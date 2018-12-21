import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Print(nn.Module):
    """
    Layer that prints the size of its input.
    Used to debug nn.Sequential
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x

class Flatten(nn.Module):
    """
    Flatten layer, to flatten convolutional layer output
    """

    def forward(self, input):
        return input.view(input.size(0), -1)

class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, memory_size=128):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        self.image_embedding_size = 128

        self.memory_size = memory_size

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(),

            Flatten(),

            nn.Linear(32 * 7 * 5, self.image_embedding_size),
            nn.LeakyReLU()
        )

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.GRUCell(self.image_embedding_size, self.memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            memory = self.memory_rnn(x, memory)
            embedding = memory
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
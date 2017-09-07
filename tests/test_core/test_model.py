from cogitare import Model


class Model1(Model):

    def forward(self, x):
        return x * 3

    def loss(self, x):
        pass

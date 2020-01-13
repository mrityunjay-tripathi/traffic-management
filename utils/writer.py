from torch.utils.tensorboard import SummaryWriter


def log(*args):
    iteration, loss, accuracy = args
    writer = SummaryWriter()
    writer.add_scalar("Loss", loss, iteration)
    writer.add_scalar("Accuracy", accuracy, iteration)


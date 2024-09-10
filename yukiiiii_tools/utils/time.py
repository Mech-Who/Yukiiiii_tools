import time
from datetime import timedelta


def sec_to_min(seconds):

    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60

    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)

    return '{}:{}'.format(minutes, seconds_remaining)


def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))


def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):

    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining

    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
        sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))

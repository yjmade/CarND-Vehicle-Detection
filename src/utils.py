# -*- coding: utf-8 -*-
import contextlib
import datetime


@contextlib.contextmanager
def log_time(*text):
    print(*text, end="...", flush=True)
    start_time = datetime.datetime.now()
    yield
    print("done in %.2fs" % (datetime.datetime.now() - start_time).total_seconds())


def print_loop(iter, *text):
    for i, item in enumerate(iter):
        print('\r>> ', *text, end=" " + str(i), flush=True)
        yield item

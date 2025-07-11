
import time
import datetime
import argparse

parser = argparse.ArgumentParser(description='rest')
parser.add_argument("--time", type=int, help="rest time in second")
args = parser.parse_args()

now = datetime.datetime.now()
print(f"Don't Knock! I'm sleep for {args.time} seconds. (@ {now})")

time.sleep(args.time)

now = datetime.datetime.now()
print(f"Hello, I woke up. (@ {now})")
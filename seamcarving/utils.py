import sys

PROGRESS_BAR = True

def cli_progress_bar(i, end_val, bar_length=20):
	if PROGRESS_BAR:
	  percent = float(i) / end_val
	  hashes = '#' * int(round(percent * bar_length))
	  spaces = ' ' * (bar_length - len(hashes))
	  sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
	  sys.stdout.flush()

def cli_progress_bar_end():
	if PROGRESS_BAR:
	  cli_progress_bar(1, 1)
	  print ''

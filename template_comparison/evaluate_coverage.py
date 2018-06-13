from __future__ import print_function
import argparse
import sys  # for commanad line
import os   # for file paths
import re 
import itertools
import cPickle as pickle 
import numpy as np

def main():
	'''Read reactions'''
	global v

	found_counts = []
	times = []
	candidate_counts = []

	# Read from file
	print('Reading from file')
	line_no = 0
	while True:
		try:
			(index, reaction_smiles, candidates, time, found_count) = pickle.load(candidates_fid)
			found_counts.append(found_count)
			times.append(time)
			candidate_counts.append(
				np.array([c_count for (c_smiles, c_count) in candidates])
			)
			line_no += 1
			if line_no % 100 == 0:
				print('done {}'.format(line_no))
		except EOFError:
			break

	# Report
	N = len(times)
	print('Found {} reactions in file'.format(N))

	# Convert to numpy
	times = np.array(times)
	found_counts = np.array(found_counts)

	with open(outfile, 'w') as outfile_fid:
		outfile_fid.write('Summary statistics for {} examples\n'.format(N))
		outfile_fid.write('Average total time per reaction example: {:.3f} examples\n'.format(
			np.mean(times)
		))
		outfile_fid.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
				'mincount', 'coverage rate', 'mean num. candidates', 'std', 'median',
				'mean time', 'std', 'median'
		))
		for mincount in [5000, 2500, 1000, 500, 250, 100, 75, 50, 25, 20, 15, 10, 5]:
			print('For mincount {}'.format(mincount))
			successfully_found = np.sum(found_counts >= mincount)
			found_times = times[found_counts >= mincount]
			num_candidates = []
			for i in range(N):
				num_candidates.append(
					np.sum(candidate_counts[i] >= mincount)
				)
			num_candidates = np.array(num_candidates)

			to_log = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
				mincount, 
				successfully_found / float(N), 
				np.mean(num_candidates), 
				np.std(num_candidates), 
				np.median(num_candidates),
				np.mean(found_times),
				np.std(found_times),
				np.median(found_times),
			)

			print(to_log)
			outfile_fid.write(to_log)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--infile', type = str, help = 'File to read candidates from')
	args = parser.parse_args()

	infile = args.infile
	outfile = 'coverage.tdf'

	with open(infile, 'rb') as candidates_fid:
		main()

#!/usr/bin/env python
import sys,os,shutil,glob,subprocess
from argparse import ArgumentParser
# Configure Argument Parser section
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle
parser = ArgumentParser(description="Prepare flat submission folder from LaTeX input")
parser.add_argument("fileHandle",
                    help="Main LaTeX file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
args = parser.parse_args()
# Create list of graphics files to copy
all_included_files_with_paths = []
for line in args.fileHandle:
    if 'includegraphics' in line:
        if '%' in line and line.find('%') < line.find('includegraphics'):
            continue
        all_included_files_with_paths.append(line.split("{",1)[1].split("}",1)[0]);
# Create submission path
path = os.getcwd()
path_submission = os.path.join(path, 'submission')
if os.path.exists(path_submission):
    sys.exit("Sumbission dir found! Remove it and rerun script!")
os.mkdir(path_submission)
# Copy all files needed
for file in all_included_files_with_paths:
    shutil.copy(os.path.join(path, file+'.pdf'), path_submission)
for fname in glob.iglob('*.bst'):
    shutil.copy(fname, path_submission)
for fname in glob.iglob('*.rtx'):
    shutil.copy(fname, path_submission)
for fname in glob.iglob('*.sty'):
    shutil.copy(fname, path_submission)
for fname in glob.iglob('*.cls'):
    shutil.copy(fname, path_submission)
shutil.copy(os.path.join(path,args.fileHandle.name)[:-3]+'bib', path_submission)
for fname in glob.iglob('../scripts/compile*.sh'):
    shutil.copy(fname, path_submission)

new_main_file = os.path.join(path_submission, args.fileHandle.name)
with open(new_main_file, 'w') as file_:
    args.fileHandle.seek(0)
    for line in args.fileHandle:
        if ('\usepackage[pdftex,unicode,colorlinks, citecolor=blue,%\n'==line) or\
           ('filecolor=black, linkcolor=blue, urlcolor=black]{hyperref}\n'==line) or\
           ('\usepackage[figure,table]{hypcap} %links should lead to the begining\n'==line):
            file_.write('%'+line)            
            continue
        if not 'includegraphics' in line:
            file_.write(line)
            continue
        if '%' in line and line.find('%') < line.find('includegraphics'):
            file_.write(line)
            continue
        current_path_to_image = line.split("{",1)[1].split("}",1)[0]
        new_path_to_image = current_path_to_image.split("/")[-1]
        file_.write(line.replace(current_path_to_image, new_path_to_image))

# Convert pdf to eps (only Linux)
try: 
    subprocess.call(['pdftops', '-v'])
except: 
    sys.exit("No pdftops executable")

os.chdir(path_submission)
for fname in glob.iglob('*.pdf'):
    subprocess.call(["pdftops", "-f", "1", "-l", "1", "-eps", fname, fname[:-3]+'eps'])
###pdftops  -f 1 -l 1 -eps "file.pdf" "file2.eps" 

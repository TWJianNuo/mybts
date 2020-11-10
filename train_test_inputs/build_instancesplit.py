import os
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

refsplit = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full_withins/{}_files.txt'
srcsplit = 'eigen_{}_files_with_gt_modified.txt'
dstsplit = 'eigen_{}_files_with_instance.txt'

processset = ['train', 'test']
for split in processset:
    refentries = readlines(refsplit.format(split))
    srcentries = readlines(srcsplit.format(split))
    dstentries = list()
    for srcentry in srcentries:
        entry = srcentry.split(' ')[0]
        tolookup1 = "{} {} {}".format("{}/{}".format(entry.split('/')[0], entry.split('/')[1]), entry.split('/')[4].split('.')[0], 'l')
        if tolookup1 in refentries:
            dstentries.append(srcentry)

    with open(dstsplit.format(split), "w") as text_file:
        for entry in dstentries:
            text_file.write(entry + '\n')


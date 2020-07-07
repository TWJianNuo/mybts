import os
root = '../train_test_inputs'
splits = ['eigen_test_files_with_gt.txt', 'eigen_train_files_with_gt.txt']
wsplits = {'eigen_test_files_with_gt.txt':'eigen_test_files_with_gt_modified.txt', 'eigen_train_files_with_gt.txt':'eigen_train_files_with_gt_modified.txt'}
for split in splits:
    fpath = os.path.join(root, split)
    f = open(fpath, 'r')
    entries = f.readlines()

    processed_entries = list()
    for entry in entries:
        comps = entry.split(' ')

        testentry = comps[1]
        if testentry != 'None':
            testcomps = testentry.split('/')
            modified_test_entry = testcomps[0][0:10] + '/' + testcomps[0] + '/' + testcomps[3] + '/' + testcomps[4]

            processed_entries.append(comps[0] + ' ' + modified_test_entry + ' ' + comps[2])
        else:
            processed_entries.append(entry)
    fw = open(os.path.join(root, wsplits[split]), 'w')
    for entry in processed_entries:
        fw.write(entry)
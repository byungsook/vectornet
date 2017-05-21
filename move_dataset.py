import os.path
import shutil

os.chdir('/home/kimby/dev/vectornet')

data_dir = 'data/qdraw_stitches_128'
output = 'data/qdraw_stitches_128_test'
if not os.path.exists(output):
    os.mkdir(output)
with open(os.path.join(data_dir,'test.txt'), 'r') as f:
    count = 0
    while True:
        line = f.readline()
        if not line: break
        file = line.rstrip()
        file_path = os.path.join(data_dir, file)
        output_path = os.path.join(output, file)
        shutil.copyfile(file_path, output_path)

        count += 1
        if count >= 100: break

print('Done')

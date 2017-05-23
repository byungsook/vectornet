import os.path
import shutil

os.chdir('/home/kimby/dev/vectornet')

data_dir = 'data/qdraw_mix_128'
output = 'data/qdraw_mix_128_test'
if not os.path.exists(output):
    os.mkdir(output)

# with open(os.path.join(data_dir,'test.txt'), 'r') as f:
#     count = 0
#     while True:
#         line = f.readline()
#         if not line: break
#         file = line.rstrip()
#         file_path = os.path.join(data_dir, file)
#         output_path = os.path.join(output, file)
#         shutil.copyfile(file_path, output_path)

#         count += 1
#         if count >= 100: break

ff = open(os.path.join(output,'test.txt'), 'w')

for data_dir in ['qdraw_cat_128',
                 'qdraw_baseball_128',
                 'qdraw_chandelier_128',
                 'qdraw_elephant_128']:
    with open(os.path.join('data', data_dir,'test.txt'), 'r') as f:
        count = 0
        while True:
            line = f.readline()
            if not line: break
            file = line.rstrip()
            file_path = os.path.join('..', data_dir, file)
            ff.write(file_path+'\n')
            output_path = os.path.join(output, file)
            shutil.copyfile(os.path.join('data', data_dir, file), output_path)

            count += 1
            if count >= 25: break

# ff.close()
print('Done')

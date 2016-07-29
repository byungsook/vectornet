# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""A binary to train Vectornet on the Sketch data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('vectornet'):
        working_path = current_path + '/vectornet'
        os.chdir(working_path)
        
    # create log directory    
    FLAGS.log_dir += datetime.now().isoformat().replace(':', '-')
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # dataset = VectornetDataset()
    dataset = DataSet(DATA_PATH, PATCH_H, PATCH_W)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()
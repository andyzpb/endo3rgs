import os
import subprocess
import glob
import sys
import argparse
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, message):
        self.log.write(message)     
    def flush(self):
        self.log.flush()

def get_args_parser():
    parser = argparse.ArgumentParser('Endo3R eval', add_help=False)
    parser.add_argument('--data_root', type=str, default='/home/medicalteam/remote_126/SCARED/test/', help='Path to input data')
    parser.add_argument('--data_type', type=str, default='scared', help='Type of validation dataset')
    parser.add_argument('--ckpt_path', type=str, default='../dynamic_endo_depth/checkpoints/selfdep_512_best.pth', help='ckpt path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--conf_thresh', type=float, default=1e-3, help='confidence threshold')
    parser.add_argument('--kf_every', type=int, default=1, help='map every kf_every frames')
    parser.add_argument('--resolution', type=int, default=320, help='map every kf_every frames')
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    root = args.data_root
    datasets_dir = sorted(glob.glob(root + "*"))

    model_path = args.ckpt_path
    eval_single_scene = False
    if args.data_type == 'scared':
        for dir in datasets_dir:
            keyframes_dir = sorted(glob.glob(dir + "/*"))
            if eval_single_scene:
                all_scenes = [root + 'dataset_9/keyframe_1']
            else:
                all_scenes = keyframes_dir
            for key in all_scenes:
                print("To reconstruct scene: ", key.split('/')[-2] + '_' + key.split('/')[-1])
                
                seq_path = os.path.join(key, "data/left_rectified/")
                output_path = "./outputs/" + model_path.split('/')[-1] + '/' + \
                    key.split('/')[-2] + '_' + key.split('/')[-1]
                
                args_list = ['python', 'demo.py', '--kf_every', str(args.kf_every),
                        '--ckpt_path', model_path, '--demo_path', seq_path,
                        '--save_path', output_path, '--resolution', str(args.resolution), '--save_result']
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '0'

                print(args_list)
                subprocess.run(args_list, env=env)
                if eval_single_scene:
                    break
            
            if eval_single_scene:
                break
            print("Evaluation for all the SCARED scenes: ")
            args_list = ['python', 'eval/depth_evaluation.py','--data_root', args.data_root,
                        '--output_path', './outputs/' + model_path.split('/')[-1]]
            print(args_list)
            subprocess.run(args_list)
    else:
        for dir in datasets_dir:
            seq_path = os.path.join(dir, "cropped01/")
            print("To reconstruct the Hamlyn scene: ", dir.split('/')[-1])
            output_path = "./outputs/" + model_path.split('/')[-1] + '_hamlyn_' + '/' + \
                dir.split('/')[-1]
                
            args_list = ['python', 'demo.py', '--kf_every', str(args.kf_every), '--data_type', 'hamlyn'
                        '--ckpt_path', model_path, '--demo_path', seq_path,
                        '--save_path', output_path, '--resolution', str(args.resolution)]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            subprocess.run(args_list, env=env)
        print("Evaluation for all the Hamlyn scenes: ")
        args_list = ['python', 'eval/depth_evaluation.py','--data_root', args.data_root, '--data_type', 'hamlyn'
                    '--output_path', './outputs/' + model_path.split('/')[-1]]
        print(args_list)
        subprocess.run(args_list)
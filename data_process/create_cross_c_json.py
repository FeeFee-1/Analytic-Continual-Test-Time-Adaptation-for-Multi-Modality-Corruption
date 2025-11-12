import os
import json
import random
import argparse
import copy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clean-path', type=str, default='/xlearning/mouxing/workspace/TTA/READ/_code_clean/json_csv_files/vgg/clean/severity_0.json')
parser.add_argument('--video-c-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/image_mulframe_test-C")
parser.add_argument('--audio-c-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/audio_test-C")
parser.add_argument('--video-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/image_mulframe_test")
parser.add_argument('--audio-path', type=str, default="/xlearning/mouxing/dataset/ImageAudio/VGGSound/audio_test")
parser.add_argument('--corruption', nargs='*', default=['all'])
args = parser.parse_args()

with open(args.clean_path, 'r') as f:
    data = json.load(f)

dic_list = data['data']
tmp_dic_list = copy.deepcopy(dic_list)

severity_list = range(1, 6)
if args.corruption[0] == 'all':
    corruption_list = [
    ('1','clean','gaussian_noise'),
    ('2','clean','shot_noise'),
    ('3','gaussian_noise','clean'),
    ('4','clean','impulse_noise'),
    ('5','clean','defocus_blur'),
    ('6','traffic','clean'), 
    ('7','clean','glass_blur'),
    ('8','clean','motion_blur'),
    ('9','crowd','clean'), 
    ('10','clean','zoom_blur'),
    ('11','clean','snow'),
    ('12','clean','frost'),
    ('13','rain','clean'),
    ('14','clean','fog'),
    ('15','clean','brightness'),
    ('16','thunder','clean'),
    ('17','clean','contrast'),
    ('18','clean','elastic_transform'),
    ('19','wind','clean'),
    ('20','clean','pixelate'),
    ('21','clean','jpeg_compression')
    ]
else:
    corruption_list = args.corruption

mixed_method_severity_list = []
for i,corruption_audio,corruption_video in corruption_list:
    mixed_severity_list = []
    for severity in severity_list:
        save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'cross')

        if not os.path.exists(os.path.join(save_path, i)):
            os.makedirs(os.path.join(save_path, i))
        dic_list = []
        for dic in tmp_dic_list:
            if corruption_audio == 'clean':
                new_dic = {
                    "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                    "wav": os.path.join(args.audio_path, '{}.wav'.format(dic.get("video_id"))),
                    "video_path": os.path.join(args.video_c_path, '{}/severity_{}/'.format(corruption_video, severity)),
                    "labels": dic.get("labels")
                }
            elif corruption_video == 'clean':
                new_dic = {
                    "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                    "wav": os.path.join(args.audio_c_path, corruption_audio, 'severity_{}'.format(severity), '{}.wav'.format(dic.get("video_id"))),
                    "video_path": args.video_path,
                    "labels": dic.get("labels")
                }
            dic_list.append(new_dic)
        print(len(dic_list))
        random.shuffle(dic_list)
        new_json = {"data": dic_list}
        with open(os.path.join(save_path, i, 'severity_{}.json'.format(severity)), "w") as file1:
            json.dump(new_json, file1, indent=1)


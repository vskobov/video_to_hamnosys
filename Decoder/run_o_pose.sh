PATH=$1
in_path=$PATH"/All_Frames/"
out_path=$PATH"/All_Keys/"

./build/examples/openpose/openpose.bin --image_dir $in_path --hand --write_json $out_path --display 0 --render_pose 0 --hand_render -1  --face_render -1 --net_resolution "-1x704" --num_gpu 2 --num_gpu_start 0 --model_pose COCO --number_people_max 1


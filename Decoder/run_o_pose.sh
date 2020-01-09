FILES=/home/Skobov_Victor/sample_dev_set_mix/*/
#FILES=/itigo/Uploads/SKOBOV_Victor/sample_dev_set_mix/*/
for f in $FILES
do
  echo "Processing $f sign..."
  frames_path="$(dirname $f})"
  #echo "$frames_path"
  dir="$(basename ${frames_path})"
  #echo "$dir"

  #./build/examples/openpose/openpose.bin --video $path$dir/$name.mp4 --hand --write_json $path$dir/$video_name --display 0 --write_video $path$dir/$video_name.mp4 --face_render 0 --hand --net_resolution "-1x512" --num_gpu 2 --num_gpu_start 0 --model_pose COCO --number_people_max 1
  ./build/examples/openpose/openpose.bin --image_dir $frames_path/frames_input_sign_h_conf_or_loc_$dir/ --hand --write_json $frames_path/keys_$dir --display 0 --render_pose 0 --hand_render -1 --face_render -1 --net_resolution "-1x704" --num_gpu 2 --num_gpu_start 0 --model_pose COCO --number_people_max 1

done

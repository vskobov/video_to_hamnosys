PATH=$1

FILES=$PATH/*/*.sigml

for f in $FILES
do
  echo "Processing $(basename ${f%.*}) sign..."

  name="$(basename ${f%.*})"
  video_path="$(dirname ${f})"
  dir="$(basename ${video_path})"
  video_name="frames_${name}"
  all_frames="$(dirname $video_path)/All_Frames/"

  java -Djava.library.path=$2/jas-libs  -Dlog4j.configurationFile=$2/../log4j2.xml -jar $2/JASApp.jar -session file:$2/../SiGMLPlayer -ja.version.tag=loc2018 -sigml.base.uri=file:$2/sigml/ -sigml.file=file:$f -ja.remote.base.url=file:$2/../../loc2018/ -do.auto.video=True -video.base.uri=file:$all_frames -video.file=$dir -do.auto.quit=False
  sleep 4
  
done

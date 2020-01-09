FILES=/Users/vskobov/Desktop/jas/loc2018/JASApp/sample_dev_set_hand_conf_wrs_5/*/*.sigml
for f in $FILES
do
  echo "Processing $f sign..."
  #echo $(basename ${f%.*})
  name="$(basename ${f%.*})"
  dir=${name: -1}
  video_name="input_video_${name: -1}"
  #echo "$dir"

  # take action on each sign file. $f store current sign name
  #cat $f
  java -Djava.library.path=jas-libs  -Dlog4j.configurationFile=../log4j2.xml -jar JASApp.jar -session file:../SiGMLPlayer -ja.version.tag=loc2018 -sigml.base.uri=file:/Users/vskobov/Desktop/jas/loc2018/JASApp/sigml/ -sigml.file=file:$f -ja.remote.base.url=file:../../loc2018/ -do.auto.video=True -video.base.uri=file:/Users/vskobov/Desktop/jas/loc2018/JASApp/sample_dev_set_hand_conf_wrs_5/$dir/ -video.file=$video_name -do.auto.quit=True
  #java -Djava.library.path=jas-libs  -Dlog4j.configurationFile=../log4j2.xml -jar JASApp.jar -session file:../SiGMLPlayer -ja.version.tag=loc2018 -sigml.base.uri=file:/Users/vskobov/Desktop/jas/loc2018/JASApp/sigml/ -sigml.file=file:$f -ja.remote.base.url=file:../../loc2018/ -do.auto.video=True -video.base.uri=file: -video.file=$(basename ${f%.*}) -do.auto.quit=True
done
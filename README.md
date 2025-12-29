# cv-nst-project
# 使用real_time需要自行下载coco数据集。运行指令：
wget http://images.cocodataset.org/zips/train2017.zip
mkdir -p data/coco
unzip train2017.zip -d data/coco/

实验中设置了style_weight为5e7,1e7和5e6。5e7的效果相对更好，其它两个值都出现了风格不连续和异常颜色的问题。
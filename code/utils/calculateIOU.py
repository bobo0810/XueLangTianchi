
class Judge_Much_IOU():
    # 现在通过计算iou的方法，来切出有瑕疵里面的无瑕疵区域
	# 其中one_x, one_y，two_x, two_y 分别表示 两个矩形框的 中心点
    def calcIOU(self,one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
        if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
            lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
            lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
            rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
            rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
            inter_w = abs(rd_x_inter - lu_x_inter)
            inter_h = abs(lu_y_inter - rd_y_inter)
            inter_square = inter_w * inter_h
            union_square = (one_w * one_h) + (two_w * two_h) - inter_square
            calcIOU = inter_square / union_square * 1.0
        else:
            calcIOU=0
        return calcIOU

    # box_list[[1,2,3,4],[1,2,3,4]]  this_box_list [xmin,ymin,xmax,ymax]
    def judge_much_IOU(self,box_list,this_box_list):
        for onelist in box_list:
            xmin_ ,ymin_ ,xmax_ ,ymax_ = onelist
            xmin_ ,ymin_ ,xmax_ ,ymax_ = int(xmin_) ,int(ymin_) ,int(xmax_) ,int(ymax_)
            xmin ,ymin ,xmax ,ymax = this_box_list
            one_x, one_y, one_w, one_h = int((xmin_ + xmax_)/2),int((ymin_ + ymax_)/2),xmax_ - xmin_ ,ymax_ - ymin_
            two_x, two_y, two_w, two_h = int((xmin + xmax)/2) ,int((ymin + ymax)/2), xmax-xmin, ymax-ymin
            result = self.calcIOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h)
            if result is not 0:
                return 0 # 表示有交叉，这样的不符合
        return 1 # 表示ok
judge_much_IOU=Judge_Much_IOU()
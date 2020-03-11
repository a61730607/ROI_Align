# ROI_Align  TEST Pro

较好的中文详解 ：http://blog.leanote.com/post/afanti.deng@gmail.com/b5f4f526490b
                https://leanote.com/api/file/getImage?fileId=5a168b96ab6441421e0026bd


zoukai 版本 ：
    设置 插值次数 ：  interpolate_times  实现取 bin_size 块 内的 interpolate_times和中心点   最大值pool 方式
    正向：
            利用 argmax_data 存储 每个 roi 点对应的四个坐标值
            利用 w_data 存储 四个对应点的权重
    反向：
            每个bottom_diff  是相关联的 top_diff 的加权求和

Bharat Singh  psroi 版本
    写死的 4中心位置插值   均值 pool 方式
    正向：
        pw+0.25  Y移动 1/4 个 bin_size_w  pw+0.75 Y移动 3/4 个 bin_size_w   刚好是bin 块 的四个采样点
        先用pw，ph 和bin_size_w，bin_size_h 得到4个float 点相关点  再对 每个提取四个关键点 做 插值  mean
    
        pxmax = min(max(roi_start_w + static_cast<Dtype>(pw + 0.75) * bin_size_w, 0.001), width - 1.001);
        pymax = min(max(roi_start_h + static_cast<Dtype>(ph + 0.75) * bin_size_h, 0.001), height - 1.001);
        pxmin = min(max(roi_start_w + static_cast<Dtype>(pw + 0.25) * bin_size_w, 0.001), width - 1.001);
        pymin = min(max(roi_start_h + static_cast<Dtype>(ph + 0.25) * bin_size_h, 0.001), height - 1.001);
        
        多做插值：
            top_data[index] = out_sum/4;
Bharat Singh deformable psroi 版本


priv -caffe  roi align   theano  版本  https://github.com/Ignotus/theano-roi-align/blob/master/roi_align.cu
   正向：
         先用pw，ph 和bin_size_w，bin_size_h 得到4个float 点相关点  在对里面所有点提取 关键点  做插值    最大值pool 方式
         
         Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
         Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
         Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
         Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;


mxnet 实现  https://github.com/SidHard/mxnet/blob/f95832d73cf3f1413ce1c77fbef0f295a7fbc57f/src/operator/roi_align.cu

        四点采样 非bin_size的中心点 如图    max pooling 方式 插值
        argmax_x 记录该点 的 maxidx_x maxidx_y

新增 Facebook官方实现 caffe2 

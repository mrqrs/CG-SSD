# CG-SSD
Corner Guided Single Stage 3D Object Detector from LiDAR Point Cloud

Ruiqi Ma, Chi Chen, Bisheng Yang, Deren Li, Haiping Wang, Yangzi Cong, Zongtian Hu

State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University
Engineering Research Center of Space-Time Data Capturing and Smart Application, the Ministry of Education of P.R.C., China
>This is the official implementation for ["Corner Guided Single Stage 3D Object Detector from LiDAR Point Cloud"](https://arxiv.org/abs/2202.11868). The code will be released once the paper is accepted.
# CG-SSD
1. Result on ONCE testing set
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-9wq8" rowspan="2">Model</th>
        <th class="tg-9wq8" colspan="3">AP@50</th>
        <th class="tg-9wq8" rowspan="2">mAP</th>
    </tr>
    <tr>
        <th class="tg-9wq8">Vehicle</th>
        <th class="tg-9wq8">Pedestrian</th>
        <th class="tg-9wq8">Cyclist</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-9wq8">PointRCNN</td>
        <td class="tg-9wq8">   52.00   </td>
        <td class="tg-9wq8">   8.73   </td>
        <td class="tg-9wq8">   34.02   </td>
        <td class="tg-9wq8">   31.58   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">PointPillar</td>
        <td class="tg-9wq8">   69.52   </td>
        <td class="tg-9wq8">   17.28   </td>
        <td class="tg-9wq8">   49.63   </td>
        <td class="tg-9wq8">   45.47   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">SECOND</td>
        <td class="tg-9wq8">   69.71   </td>
        <td class="tg-9wq8">   26.09   </td>
        <td class="tg-9wq8">   59.92   </td>
        <td class="tg-9wq8">   51.90   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">PV-RCNN</td>
        <td class="tg-9wq8"> 76.98 </td>
        <td class="tg-9wq8">   22.66   </td>
        <td class="tg-9wq8">   61.93   </td>
        <td class="tg-9wq8">   53.85   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">CenterPoint</td>
        <td class="tg-9wq8">   66.35   </td>
        <td class="tg-9wq8">   51.80   </td>
        <td class="tg-9wq8">   65.57   </td>
        <td class="tg-9wq8">   61.24   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">CG-SSD</td>
        <td class="tg-9wq8">   68.00   </td>
        <td class="tg-9wq8">   52.81   </td>
        <td class="tg-9wq8">   67.50   </td>
        <td class="tg-9wq8">   62.77   </td>
    </tr>
    </tbody>
    </table>
2. Result on ONCE Validation Set
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-9wq8" rowspan="2">Model</th>
        <th class="tg-9wq8" colspan="4">Vehicle(AP@50)</th>
        <th class="tg-9wq8" colspan="4">Pedestrian(AP@50)</th>
        <th class="tg-9wq8" colspan="4">Cyclist(AP@50)</th>
        <th class="tg-9wq8" rowspan="2">mAP</th>
    </tr>
    <tr>
        <th class="tg-9wq8">overall</th>
        <th class="tg-9wq8">0-30m</th>
        <th class="tg-9wq8">30-50m</th>
        <th class="tg-9wq8">50m-inf</th>
        <th class="tg-9wq8">overall</th>
        <th class="tg-9wq8">0-30m</th>
        <th class="tg-9wq8">30-50m</th>
        <th class="tg-9wq8">50m-inf</th>
        <th class="tg-9wq8">overall</th>
        <th class="tg-9wq8">0-30m</th>
        <th class="tg-9wq8">30-50m</th>
        <th class="tg-9wq8">50m-inf</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-9wq8">PointRCNN</td>
        <td class="tg-9wq8">   52.09   </td>
        <td class="tg-9wq8">   74.45   </td>
        <td class="tg-9wq8">   40.89   </td>
        <td class="tg-9wq8">   16.81   </td>
        <td class="tg-9wq8">   4.28   </td>
        <td class="tg-9wq8">   6.17   </td>
        <td class="tg-9wq8">   2.40   </td>
        <td class="tg-9wq8">   0.91   </td>
        <td class="tg-9wq8">   29.84   </td>
        <td class="tg-9wq8">   46.03   </td>
        <td class="tg-9wq8">   20.94   </td>
        <td class="tg-9wq8">   5.46   </td>
        <td class="tg-9wq8">   28.74   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">PointPillars</td>
        <td class="tg-9wq8">   68.57   </td>
        <td class="tg-9wq8">   80.86   </td>
        <td class="tg-9wq8">   62.07   </td>
        <td class="tg-9wq8">   47.04   </td>
        <td class="tg-9wq8">   17.63   </td>
        <td class="tg-9wq8">   19.74   </td>
        <td class="tg-9wq8">   15.15   </td>
        <td class="tg-9wq8">   10.23   </td>
        <td class="tg-9wq8">   46.81   </td>
        <td class="tg-9wq8">   58.33   </td>
        <td class="tg-9wq8">   40.32   </td>
        <td class="tg-9wq8">   25.86   </td>
        <td class="tg-9wq8">   44.34   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">SECOND</td>
        <td class="tg-9wq8">   71.19   </td>
        <td class="tg-9wq8">   84.04   </td>
        <td class="tg-9wq8">   63.02   </td>
        <td class="tg-9wq8">   47.25   </td>
        <td class="tg-9wq8">   26.44   </td>
        <td class="tg-9wq8">   29.33   </td>
        <td class="tg-9wq8">   24.05   </td>
        <td class="tg-9wq8">   18.05   </td>
        <td class="tg-9wq8">   58.04   </td>
        <td class="tg-9wq8">   69.96   </td>
        <td class="tg-9wq8">   52.43   </td>
        <td class="tg-9wq8">   34.61   </td>
        <td class="tg-9wq8">   51.89   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">PV-RCNN</td>
        <td class="tg-9wq8">   77.77  </td>
        <td class="tg-9wq8">   89.39   </td>
        <td class="tg-9wq8">   72.55   </td>
        <td class="tg-9wq8">   58.64   </td>
        <td class="tg-9wq8">   23.50   </td>
        <td class="tg-9wq8">   25.61   </td>
        <td class="tg-9wq8">   22.84   </td>
        <td class="tg-9wq8">   17.27   </td>
        <td class="tg-9wq8">   59.37   </td>
        <td class="tg-9wq8">   71.66   </td>
        <td class="tg-9wq8">   52.58   </td>
        <td class="tg-9wq8">   36.17   </td>
        <td class="tg-9wq8">   53.55   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">CenterPoint</td>
        <td class="tg-9wq8">   66.79   </td>
        <td class="tg-9wq8">   80.10   </td>
        <td class="tg-9wq8">   59.55   </td>
        <td class="tg-9wq8">   43.39   </td>
        <td class="tg-9wq8">   49.90   </td>
        <td class="tg-9wq8">   56.24   </td>
        <td class="tg-9wq8">   42.61   </td>
        <td class="tg-9wq8">   26.27   </td>
        <td class="tg-9wq8">   63.45   </td>
        <td class="tg-9wq8">   74.28   </td>
        <td class="tg-9wq8">   57.94   </td>
        <td class="tg-9wq8">   41.48   </td>
        <td class="tg-9wq8">   60.05   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">CG-SSD</td>
        <td class="tg-9wq8">   67.60   </td>
        <td class="tg-9wq8">   80.22   </td>
        <td class="tg-9wq8">   61.23   </td>
        <td class="tg-9wq8">   44.77   </td>
        <td class="tg-9wq8">   51.50   </td>
        <td class="tg-9wq8">   58.72   </td>
        <td class="tg-9wq8">   43.36   </td>
        <td class="tg-9wq8">   27.76   </td>
        <td class="tg-9wq8">   65.79   </td>
        <td class="tg-9wq8">   76.27   </td>
        <td class="tg-9wq8">   60.84   </td>
        <td class="tg-9wq8">   43.35   </td>
        <td class="tg-9wq8">   61.63   </td>
    </tr>
    </tbody>
    </table>


# PLGU-IN

1. Result on ONCE validation set
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-9wq8" rowspan="2">Model</th>
        <th class="tg-9wq8" colspan="3">AP@50</th>
        <th class="tg-9wq8" rowspan="2">mAP</th>
    </tr>
    <tr>
        <th class="tg-9wq8">   Vehicle   </th>
        <th class="tg-9wq8">   Pedestrian   </th>
        <th class="tg-9wq8">   Cyclist   </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-9wq8">PointPillars</td>
        <td class="tg-9wq8">   68.57   </td>
        <td class="tg-9wq8">   17.63   </td>
        <td class="tg-9wq8">   46.81   </td>
        <td class="tg-9wq8">   44.34   </td>
    </tr>
    <tr>
        <td class="tg-9wq8">PointPillars<br>+auxiliary module</td>
        <td class="tg-9wq8">70.45<br><span style="color:#F00">(+1.88)</td>
        <td class="tg-9wq8">18.97<br><span style="color:#F00">(+2.34)</td>
        <td class="tg-9wq8">53.71<br><span style="color:#F00">(+6.9)</td>
        <td class="tg-9wq8">47.71<br><span style="color:#F00">(+3.37)</td>
    </tr>
    <tr>
        <td class="tg-9wq8">SECOND</td>
        <td class="tg-9wq8">71.19</td>
        <td class="tg-9wq8">26.44</td>
        <td class="tg-9wq8">58.04</td>
        <td class="tg-9wq8">51.89</td>
    </tr>
    <tr>
        <td class="tg-9wq8">SECOND<br>+auxiliary module</td>
        <td class="tg-9wq8">71.10<br><span style="color:#F00">(-0.09)</td>
        <td class="tg-9wq8">38.27<br><span style="color:#F00">(+11.83)</td>
        <td class="tg-9wq8">61.89<br><span style="color:#F00">(+3.85)</td>
        <td class="tg-9wq8">57.09<br><span style="color:#F00">(+5.2)</td>
    </tr>
    <tr>
        <td class="tg-9wq8">PV-RCNN</td>
        <td class="tg-9wq8">77.77</td>
        <td class="tg-9wq8">23.50</td>
        <td class="tg-9wq8">59.37</td>
        <td class="tg-9wq8">53.55</td>
    </tr>
    <tr>
        <td class="tg-9wq8">PV-RCNN<br>+auxiliary module</td>
        <td class="tg-9wq8">79.20<br><span style="color:#F00">(+1.43)</td>
        <td class="tg-9wq8">37.19<br><span style="color:#F00">(+13.69)</td>
        <td class="tg-9wq8">65.39<br><span style="color:#F00">(+6.02)</td>
        <td class="tg-9wq8">60.60<br><span style="color:#F00">(+7.05)</td>
    </tr>
    </tbody>
    </table>
2. Result on Waymo validation set
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-nrix" rowspan="2">Model</th>
        <th class="tg-nrix" colspan="2">Vehicle/APH</th>
        <th class="tg-nrix" colspan="2">Pedestrian/APH</th>
        <th class="tg-nrix" colspan="2">Cyclist/APH</th>
    </tr>
    <tr>
        <th class="tg-nrix">L1</th>
        <th class="tg-nrix">L2</th>
        <th class="tg-nrix">L1</th>
        <th class="tg-nrix">L2</th>
        <th class="tg-nrix">L1</th>
        <th class="tg-nrix">L2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-nrix">PointPillars</td>
        <td class="tg-nrix">69.83</td>
        <td class="tg-nrix">61.64</td>
        <td class="tg-nrix">46.32</td>
        <td class="tg-nrix">40.64</td>
        <td class="tg-nrix">51.75</td>
        <td class="tg-nrix">49.80</td>
    </tr>
    <tr>
        <td class="tg-nrix">PointPillars<br>+auxiliary module</td>
        <td class="tg-nrix">71.87<br><span style="color:#F00">(+2.04)</td>
        <td class="tg-nrix">63.93<br><span style="color:#F00">(+2.29)</td>
        <td class="tg-nrix">60.55<br><span style="color:#F00">(+14.23)</td>
        <td class="tg-nrix">53.57<br><span style="color:#F00">(+12.93)</td>
        <td class="tg-nrix">63.86<br><span style="color:#F00">(+12.11)</td>
        <td class="tg-nrix">61.51<br><span style="color:#F00">(+11.71)</td>
    </tr>
    <tr>
        <td class="tg-nrix">SECOND</td>
        <td class="tg-nrix">70.34</td>
        <td class="tg-nrix">62.02</td>
        <td class="tg-nrix">54.24</td>
        <td class="tg-nrix">47.49</td>
        <td class="tg-nrix">55.62</td>
        <td class="tg-nrix">53.53</td>
    </tr>
    <tr>
        <td class="tg-nrix">SECOND<br>+auxiliary module</td>
        <td class="tg-nrix">71.65<br><span style="color:#F00">(+1.31)</td>
        <td class="tg-nrix">63.78<br><span style="color:#F00">(+1.76)</td>
        <td class="tg-nrix">58.01<br><span style="color:#F00">(+3.77)</td>
        <td class="tg-nrix">51.03<br><span style="color:#F00">(+3.54)</td>
        <td class="tg-nrix">63.31<br><span style="color:#F00">(+7.69)</td>
        <td class="tg-nrix">61.01<br><span style="color:#F00">(+7.48)</td>
    </tr>
    <tr>
        <td class="tg-nrix">PV-RCNN</td>
        <td class="tg-nrix">74.74</td>
        <td class="tg-nrix">66.80</td>
        <td class="tg-nrix">61.24</td>
        <td class="tg-nrix">53.95</td>
        <td class="tg-nrix">64.25</td>
        <td class="tg-nrix">61.82</td>
    </tr>
    <tr>
        <td class="tg-nrix">PV-RCNN<br>+auxiliary module</td>
        <td class="tg-nrix">76.53<br><span style="color:#F00">(+1.79)</td>
        <td class="tg-nrix">67.97<br><span style="color:#F00">(+1.17)</td>
        <td class="tg-nrix">66.62<br><span style="color:#F00">(+5.38)</td>
        <td class="tg-nrix">58.36<br><span style="color:#F00">(+4.41)</td>
        <td class="tg-nrix">69.63<br><span style="color:#F00">(+5.38)</td>
        <td class="tg-nrix">67.12<br><span style="color:#F00">(+5.3)</td>
    </tr>
    </tbody>
    </table>
文件说明：
	1. drow_point_line.py 绘制的是MLCTAI方法校正与未校正的预测值之间的散点图
	2. recalibration_masterPLS.py 是用来做MLCTAI与重新校准与主仪器PLS模型的对比的。
	3. SNV_affine_corn/yaopian.py 是用作计算MLCTAI与其余五种方法的对比结果的算法	
	4. SNV_affine_corn/yaopian_img.py	是用作绘制最终的MLCTAI方法与其余五种方法的预测结果分布的情况的。
	5. withStandardTest.py	用于获取最佳的标样数量
	4. draw_spectrumas.py	用于画数据集光谱图。
	
数据集筛选：
	
	
	
	药片集不适用SNV，选择第二第三个成分做，结果很好。
	玉米集有几个选择：
		全部不使用SNV
			m5-mp6 
	
	
	
标样选择规则：
	1. 选择标样方法中最小的RMSEC的标样数下的RMSEP，而不是最小的RMSEP
	2. 每次运行都要检查，不能出一点错。
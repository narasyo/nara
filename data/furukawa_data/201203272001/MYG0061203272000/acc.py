import PySGM

file_name = "MYG0061203272000.EW"  # .EWファイルのみ指定すれば良い
acc = PySGM.parse(file_name,fmt="nied")   # 自動的に .NS .UD ファイルも読み込んで，3成分記録をvectorオブジェクト（独自）に変換

acc.trend_removal()  # vectorオブジェクトでは様々な解析ができます．まずは基線補正．
acc.peak_ground_3d() # 3成分合成の最大加速度が出力されます

acc.output("MYG006201203272000.acc")  # ファイルに出力できます



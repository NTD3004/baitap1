%run src/pca_analysis.py: chạy phân tích pca

%run src/linear_regression.py --pca-components 3 --cv-folds 5: chạy mô hình hồi quy tuyến tính với thành phần pca bằng 3 và số round CV bằng 5.

%run src/knn_regression.py --pca-components 3 --k-values 3 5 7 9 11 15 20 --cv-folds 5: chạy mô hình k-NN hồi quy với thành phần pca bằng 3 và số round CV bằng 5 và siêu tham số k = {3,5,7,9,11,15,20}.

%run src/decision_tree.py --pca-components 3 --cv-folds 5 --max-depth-values 2 3 4 5 none --min-samples-leaf-values 1 5 10 20: chạy mô hình cây quyết định hồi quy với thành phần pca bằng 3 và số round CV bằng 5 và 2 siêu tham số min_samples_leaf = {1,5,10,20} và max_depth_values = {2,3,4,5,NONE}.

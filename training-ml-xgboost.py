import os,sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main(output_folder):
    #โหลดไฟล์ บรรทัดนี้จะได้ path output_folder เช่น data/output/training-set.csv
    train_file = os.path.join(output_folder,'training-set.csv')
    test_file = os.path.join(output_folder,'testing-set.csv')
    #จากนั้นเช็ค ว่า path นั้นมีอยู่มั้ยถ้าไม่มีแสดง There is no training-set or testing-set.csv
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        sys.exit('There is no training-set or testing-set.csv')
    #จากนั้นอ่าน csv จาก train_file , test_file
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    # เช็คว่า column label มีอยู่ใน df_train column หรือไม่
    if 'label' not in df_train.columns:
        sys.exit('There is no label column in dataset')

    x_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    x_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    #ใช้ xgboost model
    model = xgb.XGBClassifier(n_estimators=100, # parameter ไม่ควร hardcode และควรรับ จาก env var
    learning_rate=0.1,max_depth=6, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(x_train,y_train)

    #ทำนาย
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict))

    #save model
    model_path = os.path.join(output_folder,'xgboost-model.pkl')
    joblib.dump(model,model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("วิธีใช้: python training-ml-xgboost.py <output_folder>")
        sys.exit(1)
    output_folder = sys.argv[1]
    main(output_folder)
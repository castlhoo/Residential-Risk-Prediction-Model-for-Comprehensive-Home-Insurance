from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col

# Spark 세션 생성
spark = SparkSession.builder.appName("Apartment Crime Safety Prediction").getOrCreate()

# 1. 학습용 데이터 로드
train_data_path = "/home/username/APT_Crime_Data.csv"  # 학습용 CSV 파일 경로
train_data = spark.read.csv(train_data_path, header=True, inferSchema=True)

# 2. 필요한 열을 선택
train_data = train_data.select("Accessibility", "Surveillance", "Crime Count", "Risk Score")

# 3. `Risk Score` 값이 100인 경우 99로 조정하고, 정수로 변환
train_data = train_data.withColumn("Risk Score", when(col("Risk Score") >= 100, 99).otherwise(col("Risk Score").cast("integer")))

# 중복된 feature 컬럼이 있는지 확인하고 삭제
if "features" in train_data.columns:
    train_data = train_data.drop("features")

# 4. Feature Vector 생성
assembler = VectorAssembler(inputCols=["Accessibility", "Surveillance", "Crime Count"], outputCol="features")

# 5. 분류 모델 생성 (Random Forest Classifier)
rf = RandomForestClassifier(labelCol="Risk Score", featuresCol="features", numTrees=10)

# 6. 파이프라인 설정
pipeline = Pipeline(stages=[assembler, rf])

# 7. 모델 학습
model = pipeline.fit(train_data)

# 8. 예측용 데이터 로드 (별도의 CSV 파일)
test_data_path = "/home/username/APT_Prediction.csv"  # 예측용 CSV 파일 경로
test_data = spark.read.csv(test_data_path, header=True, inferSchema=True)

# 9. 예측용 데이터 준비 (Risk Score는 예측을 위해 제외)
test_data = test_data.select("Apartment Name", "Accessibility", "Surveillance", "Crime Count")

# 중복된 feature 컬럼이 있는지 확인하고 삭제 (테스트 데이터에 대해서도)
if "features" in test_data.columns:
    test_data = test_data.drop("features")

# 10. 예측 수행
predictions = model.transform(test_data)

# 11. 예측된 Risk Score를 기존 데이터에 추가
predictions_with_risk_score = predictions.select("Apartment Name", "Accessibility", "Surveillance", "Crime Count", col("prediction").alias("Predicted Risk Score"))

# 12. 모델 평가 (성능 평가)
# 테스트 데이터에 실제 "Risk Score"가 없으므로 이 부분은 생략합니다.
print("Unable to evaluate model accuracy: 'Risk Score' column not found in predictions.")

# 13. 예측 결과를 CSV 파일로 저장
output_path = "/home/username/APT_Prediction_with_Risk_Score.csv"
predictions_with_risk_score.write.mode("overwrite").csv(output_path, header=True)

print(f"Predicted results saved to {output_path}")

# Spark 세션 종료
spark.stop()
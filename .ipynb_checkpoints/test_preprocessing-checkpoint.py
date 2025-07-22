{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "146e978c-c55d-4e14-a713-a81e0327f409",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mTypeError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[8]\u001b[39m\u001b[32m, line 9\u001b[39m\n\u001b[32m      6\u001b[39m csv_path = \u001b[33m'\u001b[39m\u001b[33manxiety.csv\u001b[39m\u001b[33m'\u001b[39m\n\u001b[32m      8\u001b[39m \u001b[38;5;66;03m# Run the pipeline\u001b[39;00m\n\u001b[32m----> \u001b[39m\u001b[32m9\u001b[39m X_train, X_test, y_train, y_test, preprocessor, smote = \u001b[43mload_and_preprocess\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcsv_path\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m     11\u001b[39m \u001b[38;5;66;03m# Output results\u001b[39;00m\n\u001b[32m     12\u001b[39m \u001b[38;5;28mprint\u001b[39m(\u001b[33m\"\u001b[39m\u001b[33m✅ Preprocessing succeeded!\u001b[39m\u001b[33m\"\u001b[39m)\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\anxiety-risk-api\\src\\data_processing.py:33\u001b[39m, in \u001b[36mload_and_preprocess\u001b[39m\u001b[34m(csv_path, test_size, random_state)\u001b[39m\n\u001b[32m     31\u001b[39m \u001b[38;5;66;03m# 4. Build transformers\u001b[39;00m\n\u001b[32m     32\u001b[39m numeric_transformer = StandardScaler()\n\u001b[32m---> \u001b[39m\u001b[32m33\u001b[39m categorical_transformer = \u001b[43mOneHotEncoder\u001b[49m\u001b[43m(\u001b[49m\u001b[43mhandle_unknown\u001b[49m\u001b[43m=\u001b[49m\u001b[33;43m'\u001b[39;49m\u001b[33;43mignore\u001b[39;49m\u001b[33;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43msparse\u001b[49m\u001b[43m=\u001b[49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[32m     34\u001b[39m preprocessor = ColumnTransformer([\n\u001b[32m     35\u001b[39m     (\u001b[33m'\u001b[39m\u001b[33mnum\u001b[39m\u001b[33m'\u001b[39m, numeric_transformer, numeric_cols),\n\u001b[32m     36\u001b[39m     (\u001b[33m'\u001b[39m\u001b[33mcat\u001b[39m\u001b[33m'\u001b[39m, categorical_transformer, categorical_cols),\n\u001b[32m     37\u001b[39m ])\n\u001b[32m     39\u001b[39m \u001b[38;5;66;03m# 5. Apply preprocessing\u001b[39;00m\n",
      "\u001b[31mTypeError\u001b[39m: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'"
     ]
    }
   ],
   "source": [
    "# test_preprocessing.py\n",
    "print(\"Starting test…\")\n",
    "from src.data_processing import load_and_preprocess\n",
    "\n",
    "# Adjust path if needed; this assumes your CSV is at data/anxiety_data.csv\n",
    "csv_path = 'anxiety.csv'\n",
    "\n",
    "# Run the pipeline\n",
    "X_train, X_test, y_train, y_test, preprocessor, smote = load_and_preprocess(csv_path)\n",
    "\n",
    "# Output results\n",
    "print(\"✅ Preprocessing succeeded!\")\n",
    "print(f\"X_train shape: {X_train.shape}\")\n",
    "print(f\"X_test  shape: {X_test.shape}\")\n",
    "print(\"\\nClass distribution in y_train:\")\n",
    "print(y_train.value_counts())\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

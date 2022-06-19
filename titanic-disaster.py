{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "942cc6b6",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:42.395103Z",
     "iopub.status.busy": "2022-06-19T16:26:42.394621Z",
     "iopub.status.idle": "2022-06-19T16:26:42.407974Z",
     "shell.execute_reply": "2022-06-19T16:26:42.407305Z"
    },
    "papermill": {
     "duration": 0.03008,
     "end_time": "2022-06-19T16:26:42.410225",
     "exception": false,
     "start_time": "2022-06-19T16:26:42.380145",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/titanic/train.csv\n",
      "/kaggle/input/titanic/test.csv\n",
      "/kaggle/input/titanic/gender_submission.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f3abe91e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:42.434373Z",
     "iopub.status.busy": "2022-06-19T16:26:42.433341Z",
     "iopub.status.idle": "2022-06-19T16:26:43.691829Z",
     "shell.execute_reply": "2022-06-19T16:26:43.690954Z"
    },
    "papermill": {
     "duration": 1.272883,
     "end_time": "2022-06-19T16:26:43.694327",
     "exception": false,
     "start_time": "2022-06-19T16:26:42.421444",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import seaborn\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "015e74d2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.719258Z",
     "iopub.status.busy": "2022-06-19T16:26:43.718181Z",
     "iopub.status.idle": "2022-06-19T16:26:43.746668Z",
     "shell.execute_reply": "2022-06-19T16:26:43.745588Z"
    },
    "papermill": {
     "duration": 0.043562,
     "end_time": "2022-06-19T16:26:43.749237",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.705675",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_train = pd.read_csv('/kaggle/input/titanic/train.csv')\n",
    "df_test = pd.read_csv('/kaggle/input/titanic/test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0b903a1c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.773532Z",
     "iopub.status.busy": "2022-06-19T16:26:43.772850Z",
     "iopub.status.idle": "2022-06-19T16:26:43.800760Z",
     "shell.execute_reply": "2022-06-19T16:26:43.799644Z"
    },
    "papermill": {
     "duration": 0.043624,
     "end_time": "2022-06-19T16:26:43.804058",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.760434",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bee84acf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.828895Z",
     "iopub.status.busy": "2022-06-19T16:26:43.828484Z",
     "iopub.status.idle": "2022-06-19T16:26:43.842334Z",
     "shell.execute_reply": "2022-06-19T16:26:43.840819Z"
    },
    "papermill": {
     "duration": 0.028655,
     "end_time": "2022-06-19T16:26:43.844520",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.815865",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Pclass       418 non-null    int64  \n",
      " 2   Name         418 non-null    object \n",
      " 3   Sex          418 non-null    object \n",
      " 4   Age          332 non-null    float64\n",
      " 5   SibSp        418 non-null    int64  \n",
      " 6   Parch        418 non-null    int64  \n",
      " 7   Ticket       418 non-null    object \n",
      " 8   Fare         417 non-null    float64\n",
      " 9   Cabin        91 non-null     object \n",
      " 10  Embarked     418 non-null    object \n",
      "dtypes: float64(2), int64(4), object(5)\n",
      "memory usage: 36.0+ KB\n"
     ]
    }
   ],
   "source": [
    "df_test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "710687c4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.869248Z",
     "iopub.status.busy": "2022-06-19T16:26:43.868582Z",
     "iopub.status.idle": "2022-06-19T16:26:43.874929Z",
     "shell.execute_reply": "2022-06-19T16:26:43.873932Z"
    },
    "papermill": {
     "duration": 0.020849,
     "end_time": "2022-06-19T16:26:43.876986",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.856137",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train = df_train.pop('Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4719e246",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.901784Z",
     "iopub.status.busy": "2022-06-19T16:26:43.901247Z",
     "iopub.status.idle": "2022-06-19T16:26:43.923973Z",
     "shell.execute_reply": "2022-06-19T16:26:43.922987Z"
    },
    "papermill": {
     "duration": 0.038129,
     "end_time": "2022-06-19T16:26:43.926342",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.888213",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ddde7983",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.951058Z",
     "iopub.status.busy": "2022-06-19T16:26:43.950662Z",
     "iopub.status.idle": "2022-06-19T16:26:43.960969Z",
     "shell.execute_reply": "2022-06-19T16:26:43.959678Z"
    },
    "papermill": {
     "duration": 0.025411,
     "end_time": "2022-06-19T16:26:43.963210",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.937799",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#combining test and train and looking at the data\n",
    "frames = [df_test,df_train]\n",
    "df = pd.concat(frames)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7a9c1e2a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:43.988063Z",
     "iopub.status.busy": "2022-06-19T16:26:43.987661Z",
     "iopub.status.idle": "2022-06-19T16:26:44.004142Z",
     "shell.execute_reply": "2022-06-19T16:26:44.003168Z"
    },
    "papermill": {
     "duration": 0.0316,
     "end_time": "2022-06-19T16:26:44.006327",
     "exception": false,
     "start_time": "2022-06-19T16:26:43.974727",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1309 entries, 0 to 890\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  1309 non-null   int64  \n",
      " 1   Pclass       1309 non-null   int64  \n",
      " 2   Name         1309 non-null   object \n",
      " 3   Sex          1309 non-null   object \n",
      " 4   Age          1046 non-null   float64\n",
      " 5   SibSp        1309 non-null   int64  \n",
      " 6   Parch        1309 non-null   int64  \n",
      " 7   Ticket       1309 non-null   object \n",
      " 8   Fare         1308 non-null   float64\n",
      " 9   Cabin        295 non-null    object \n",
      " 10  Embarked     1307 non-null   object \n",
      "dtypes: float64(2), int64(4), object(5)\n",
      "memory usage: 122.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info() # Age,Cabin,Fare,Embarked have null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "513a6f67",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.031662Z",
     "iopub.status.busy": "2022-06-19T16:26:44.031267Z",
     "iopub.status.idle": "2022-06-19T16:26:44.453837Z",
     "shell.execute_reply": "2022-06-19T16:26:44.452833Z"
    },
    "papermill": {
     "duration": 0.438137,
     "end_time": "2022-06-19T16:26:44.456446",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.018309",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Age-Filling it based on the mean of person's first name\n",
    "df['Title'] = df['Name'].str.extract('([A-Za-z]+)\\.')\n",
    "df['Title']=df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',\n",
    "                                               'Rev','Capt','Sir','Don', 'Dona','Master'],\n",
    "                                              ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other',\n",
    "                                               'Mr','Mr','Mr', 'Mrs','Mr']\n",
    "                                             )\n",
    "#replace mean of the age based on the person's title\n",
    "for index, row in df.iterrows():\n",
    "    if np.isnan(row[\"Age\"]):\n",
    "        df.loc[index,'Age']=df[df['Title']==row['Title']]['Age'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e08cc085",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.482759Z",
     "iopub.status.busy": "2022-06-19T16:26:44.482379Z",
     "iopub.status.idle": "2022-06-19T16:26:44.487715Z",
     "shell.execute_reply": "2022-06-19T16:26:44.487013Z"
    },
    "papermill": {
     "duration": 0.02066,
     "end_time": "2022-06-19T16:26:44.489605",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.468945",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# filling fare with mean value\n",
    "mean_value=df['Fare'].mean()\n",
    "df['Fare'].fillna(value=mean_value, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "9978f98b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.515946Z",
     "iopub.status.busy": "2022-06-19T16:26:44.514782Z",
     "iopub.status.idle": "2022-06-19T16:26:44.524921Z",
     "shell.execute_reply": "2022-06-19T16:26:44.524168Z"
    },
    "papermill": {
     "duration": 0.025448,
     "end_time": "2022-06-19T16:26:44.527071",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.501623",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Encode Cabin,Embarked and Sex\n",
    "# creating instance of labelencoder\n",
    "labelencoder = LabelEncoder()\n",
    "#Assigning numerical values and storing in another column\n",
    "df['Sex_en'] = labelencoder.fit_transform(df['Sex'])\n",
    "df['Cabin_en'] = labelencoder.fit_transform(df['Cabin'])\n",
    "df['Embarked_en'] = labelencoder.fit_transform(df['Embarked'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e2132e88",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.552758Z",
     "iopub.status.busy": "2022-06-19T16:26:44.551951Z",
     "iopub.status.idle": "2022-06-19T16:26:44.558187Z",
     "shell.execute_reply": "2022-06-19T16:26:44.557484Z"
    },
    "papermill": {
     "duration": 0.021556,
     "end_time": "2022-06-19T16:26:44.560257",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.538701",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df['Family_size'] = df['Parch']+ df['SibSp'] + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "be507795",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.585971Z",
     "iopub.status.busy": "2022-06-19T16:26:44.585201Z",
     "iopub.status.idle": "2022-06-19T16:26:44.599809Z",
     "shell.execute_reply": "2022-06-19T16:26:44.598777Z"
    },
    "papermill": {
     "duration": 0.030266,
     "end_time": "2022-06-19T16:26:44.601907",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.571641",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1309 entries, 0 to 890\n",
      "Data columns (total 16 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  1309 non-null   int64  \n",
      " 1   Pclass       1309 non-null   int64  \n",
      " 2   Name         1309 non-null   object \n",
      " 3   Sex          1309 non-null   object \n",
      " 4   Age          1309 non-null   float64\n",
      " 5   SibSp        1309 non-null   int64  \n",
      " 6   Parch        1309 non-null   int64  \n",
      " 7   Ticket       1309 non-null   object \n",
      " 8   Fare         1309 non-null   float64\n",
      " 9   Cabin        295 non-null    object \n",
      " 10  Embarked     1307 non-null   object \n",
      " 11  Title        1309 non-null   object \n",
      " 12  Sex_en       1309 non-null   int64  \n",
      " 13  Cabin_en     1309 non-null   int64  \n",
      " 14  Embarked_en  1309 non-null   int64  \n",
      " 15  Family_size  1309 non-null   int64  \n",
      "dtypes: float64(2), int64(8), object(6)\n",
      "memory usage: 206.1+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "9f3e21e9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.627272Z",
     "iopub.status.busy": "2022-06-19T16:26:44.626841Z",
     "iopub.status.idle": "2022-06-19T16:26:44.636995Z",
     "shell.execute_reply": "2022-06-19T16:26:44.636196Z"
    },
    "papermill": {
     "duration": 0.025726,
     "end_time": "2022-06-19T16:26:44.639118",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.613392",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df[[\"Pclass\",\"Age\",\"Family_size\"]] = df[[\"Pclass\",\"Age\",\"Family_size\"]].astype(\"int64\")\n",
    "x_train = df[df['PassengerId'].isin(df_train['PassengerId'])]\n",
    "x_test  = df[df['PassengerId'].isin(df_test['PassengerId'])]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53104ea7",
   "metadata": {
    "papermill": {
     "duration": 0.01189,
     "end_time": "2022-06-19T16:26:44.663415",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.651525",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "EDA****"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c6faf865",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.690665Z",
     "iopub.status.busy": "2022-06-19T16:26:44.689999Z",
     "iopub.status.idle": "2022-06-19T16:26:44.696359Z",
     "shell.execute_reply": "2022-06-19T16:26:44.695341Z"
    },
    "papermill": {
     "duration": 0.022898,
     "end_time": "2022-06-19T16:26:44.699264",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.676366",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "eda = x_train\n",
    "eda['Survived'] = y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6f2846c2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.725707Z",
     "iopub.status.busy": "2022-06-19T16:26:44.725299Z",
     "iopub.status.idle": "2022-06-19T16:26:44.741093Z",
     "shell.execute_reply": "2022-06-19T16:26:44.739993Z"
    },
    "papermill": {
     "duration": 0.031765,
     "end_time": "2022-06-19T16:26:44.743908",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.712143",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 891 entries, 0 to 890\n",
      "Data columns (total 17 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Pclass       891 non-null    int64  \n",
      " 2   Name         891 non-null    object \n",
      " 3   Sex          891 non-null    object \n",
      " 4   Age          891 non-null    int64  \n",
      " 5   SibSp        891 non-null    int64  \n",
      " 6   Parch        891 non-null    int64  \n",
      " 7   Ticket       891 non-null    object \n",
      " 8   Fare         891 non-null    float64\n",
      " 9   Cabin        204 non-null    object \n",
      " 10  Embarked     889 non-null    object \n",
      " 11  Title        891 non-null    object \n",
      " 12  Sex_en       891 non-null    int64  \n",
      " 13  Cabin_en     891 non-null    int64  \n",
      " 14  Embarked_en  891 non-null    int64  \n",
      " 15  Family_size  891 non-null    int64  \n",
      " 16  Survived     891 non-null    int64  \n",
      "dtypes: float64(1), int64(10), object(6)\n",
      "memory usage: 125.3+ KB\n"
     ]
    }
   ],
   "source": [
    "eda.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "13ee053e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.770480Z",
     "iopub.status.busy": "2022-06-19T16:26:44.770043Z",
     "iopub.status.idle": "2022-06-19T16:26:44.969601Z",
     "shell.execute_reply": "2022-06-19T16:26:44.968821Z"
    },
    "papermill": {
     "duration": 0.215009,
     "end_time": "2022-06-19T16:26:44.971841",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.756832",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Pclass', ylabel='count'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAAAXJklEQVR4nO3df5BdZZ3n8ffHJEMYgyKkZUM6kqg4IxGM0iAOo5XFUpB1AzMFBGomgOAEBbZi1YwlWq6gNWyxilqKrlamUEAZQgBdWArZRUQZFcE0E34FKKKodCpKEhSMyq/43T/65NADndCBvvcm6fer6lSf85wf/T25Vf3Jc85zz0lVIUkSwEt6XYAkafthKEiSWoaCJKllKEiSWoaCJKk1udcFvBjTp0+v2bNn97oMSdqhDA4Orq+qvtHW7dChMHv2bFasWNHrMiRph5LkF1ta5+UjSVLLUJAktQwFSVJrh76nMJqnnnqKoaEhHn/88V6X8qJNnTqV/v5+pkyZ0utSJE0QO10oDA0NsdtuuzF79myS9LqcF6yq2LBhA0NDQ8yZM6fX5UiaIHa6y0ePP/44e+655w4dCABJ2HPPPXeKHo+kHcdOFwrADh8Im+0s5yFpx7FThoIk6YWZMKFw7rnnMnfuXA444ADmzZvHrbfe+qKPec0113DeeeeNQ3Uwbdq0cTmOJL0YO92N5tHccsstXHvttdx+++3ssssurF+/nieffHJM+z799NNMnjz6P9OCBQtYsGDBeJYq7ZQO/NAlvS5hmwx++sRel9AzE6KnsHbtWqZPn84uu+wCwPTp09l7772ZPXs269evB2DFihXMnz8fgHPOOYdFixZx6KGHsmjRIg455BDuueee9njz589nxYoVXHTRRZx55pk8+uij7LPPPvzpT38C4Pe//z2zZs3iqaee4qc//SlHHHEEBx54IG9729u47777AHjwwQd561vfyv7778/HPvaxLv5rSNKWTYhQeNe73sVDDz3E6173Ok4//XS+//3vP+8+q1at4jvf+Q6XXXYZCxcuZPny5cBwwKxdu5aBgYF225e//OXMmzevPe61117L4YcfzpQpU1i8eDEXXHABg4ODnH/++Zx++ukALFmyhA984APcddddzJgxowNnLUnbbkKEwrRp0xgcHGTp0qX09fWxcOFCLrrooq3us2DBAnbddVcAjjvuOK68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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=eda,x='Pclass',hue='Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "166a7c95",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:44.999605Z",
     "iopub.status.busy": "2022-06-19T16:26:44.998956Z",
     "iopub.status.idle": "2022-06-19T16:26:45.132666Z",
     "shell.execute_reply": "2022-06-19T16:26:45.131686Z"
    },
    "papermill": {
     "duration": 0.149823,
     "end_time": "2022-06-19T16:26:45.134884",
     "exception": false,
     "start_time": "2022-06-19T16:26:44.985061",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=eda,x='Sex',hue='Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "efbad276",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.162714Z",
     "iopub.status.busy": "2022-06-19T16:26:45.161727Z",
     "iopub.status.idle": "2022-06-19T16:26:45.328908Z",
     "shell.execute_reply": "2022-06-19T16:26:45.327977Z"
    },
    "papermill": {
     "duration": 0.18347,
     "end_time": "2022-06-19T16:26:45.331046",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.147576",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Embarked', ylabel='count'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=eda,x='Embarked',hue='Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3170c294",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.359032Z",
     "iopub.status.busy": "2022-06-19T16:26:45.358369Z",
     "iopub.status.idle": "2022-06-19T16:26:45.618619Z",
     "shell.execute_reply": "2022-06-19T16:26:45.617791Z"
    },
    "papermill": {
     "duration": 0.276609,
     "end_time": "2022-06-19T16:26:45.620762",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.344153",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Family_size', ylabel='count'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=eda,x='Family_size',hue='Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b04986c6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.649855Z",
     "iopub.status.busy": "2022-06-19T16:26:45.649231Z",
     "iopub.status.idle": "2022-06-19T16:26:45.662823Z",
     "shell.execute_reply": "2022-06-19T16:26:45.662078Z"
    },
    "papermill": {
     "duration": 0.0313,
     "end_time": "2022-06-19T16:26:45.665440",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.634140",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 891 entries, 0 to 890\n",
      "Data columns (total 17 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Pclass       891 non-null    int64  \n",
      " 2   Name         891 non-null    object \n",
      " 3   Sex          891 non-null    object \n",
      " 4   Age          891 non-null    int64  \n",
      " 5   SibSp        891 non-null    int64  \n",
      " 6   Parch        891 non-null    int64  \n",
      " 7   Ticket       891 non-null    object \n",
      " 8   Fare         891 non-null    float64\n",
      " 9   Cabin        204 non-null    object \n",
      " 10  Embarked     889 non-null    object \n",
      " 11  Title        891 non-null    object \n",
      " 12  Sex_en       891 non-null    int64  \n",
      " 13  Cabin_en     891 non-null    int64  \n",
      " 14  Embarked_en  891 non-null    int64  \n",
      " 15  Family_size  891 non-null    int64  \n",
      " 16  Survived     891 non-null    int64  \n",
      "dtypes: float64(1), int64(10), object(6)\n",
      "memory usage: 157.6+ KB\n"
     ]
    }
   ],
   "source": [
    "x_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5360439a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.696117Z",
     "iopub.status.busy": "2022-06-19T16:26:45.695530Z",
     "iopub.status.idle": "2022-06-19T16:26:45.708518Z",
     "shell.execute_reply": "2022-06-19T16:26:45.707582Z"
    },
    "papermill": {
     "duration": 0.031126,
     "end_time": "2022-06-19T16:26:45.711575",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.680449",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 891 entries, 0 to 890\n",
      "Data columns (total 17 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Pclass       891 non-null    int64  \n",
      " 2   Name         891 non-null    object \n",
      " 3   Sex          891 non-null    object \n",
      " 4   Age          891 non-null    int64  \n",
      " 5   SibSp        891 non-null    int64  \n",
      " 6   Parch        891 non-null    int64  \n",
      " 7   Ticket       891 non-null    object \n",
      " 8   Fare         891 non-null    float64\n",
      " 9   Cabin        204 non-null    object \n",
      " 10  Embarked     889 non-null    object \n",
      " 11  Title        891 non-null    object \n",
      " 12  Sex_en       891 non-null    int64  \n",
      " 13  Cabin_en     891 non-null    int64  \n",
      " 14  Embarked_en  891 non-null    int64  \n",
      " 15  Family_size  891 non-null    int64  \n",
      " 16  Survived     891 non-null    int64  \n",
      "dtypes: float64(1), int64(10), object(6)\n",
      "memory usage: 157.6+ KB\n"
     ]
    }
   ],
   "source": [
    "x_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "21a6288b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.742679Z",
     "iopub.status.busy": "2022-06-19T16:26:45.741595Z",
     "iopub.status.idle": "2022-06-19T16:26:45.748460Z",
     "shell.execute_reply": "2022-06-19T16:26:45.747703Z"
    },
    "papermill": {
     "duration": 0.023736,
     "end_time": "2022-06-19T16:26:45.750500",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.726764",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_train = x_train.drop(['Name','PassengerId','Ticket','Cabin','Embarked','Title','Sex','SibSp','Parch'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "fdf44d58",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.779137Z",
     "iopub.status.busy": "2022-06-19T16:26:45.778713Z",
     "iopub.status.idle": "2022-06-19T16:26:45.783758Z",
     "shell.execute_reply": "2022-06-19T16:26:45.782921Z"
    },
    "papermill": {
     "duration": 0.021816,
     "end_time": "2022-06-19T16:26:45.785670",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.763854",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train = x_train.pop('Survived')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e9a6a2ab",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.815328Z",
     "iopub.status.busy": "2022-06-19T16:26:45.814361Z",
     "iopub.status.idle": "2022-06-19T16:26:45.830724Z",
     "shell.execute_reply": "2022-06-19T16:26:45.829742Z"
    },
    "papermill": {
     "duration": 0.033284,
     "end_time": "2022-06-19T16:26:45.832750",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.799466",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_en</th>\n",
       "      <th>Cabin_en</th>\n",
       "      <th>Embarked_en</th>\n",
       "      <th>Family_size</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>22</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>106</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>26</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>70</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>35</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>2</td>\n",
       "      <td>27</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>3</td>\n",
       "      <td>23</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>0</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>1</td>\n",
       "      <td>77</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>3</td>\n",
       "      <td>32</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Age     Fare  Sex_en  Cabin_en  Embarked_en  Family_size\n",
       "0         3   22   7.2500       1       186            2            2\n",
       "1         1   38  71.2833       0       106            0            2\n",
       "2         3   26   7.9250       0       186            2            1\n",
       "3         1   35  53.1000       0        70            2            2\n",
       "4         3   35   8.0500       1       186            2            1\n",
       "..      ...  ...      ...     ...       ...          ...          ...\n",
       "886       2   27  13.0000       1       186            2            1\n",
       "887       1   19  30.0000       0        40            2            1\n",
       "888       3   23  23.4500       0       186            2            4\n",
       "889       1   26  30.0000       1        77            0            1\n",
       "890       3   32   7.7500       1       186            1            1\n",
       "\n",
       "[891 rows x 7 columns]"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9f52e828",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.862797Z",
     "iopub.status.busy": "2022-06-19T16:26:45.862364Z",
     "iopub.status.idle": "2022-06-19T16:26:45.887474Z",
     "shell.execute_reply": "2022-06-19T16:26:45.886174Z"
    },
    "papermill": {
     "duration": 0.044228,
     "end_time": "2022-06-19T16:26:45.890827",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.846599",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py:3678: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  self[col] = igetitem(value, i)\n"
     ]
    }
   ],
   "source": [
    "#Scaling\n",
    "from sklearn.preprocessing import StandardScaler,MinMaxScaler\n",
    "scale_col = ['Pclass','Age','Family_size','Fare']\n",
    "scaler = MinMaxScaler()\n",
    "scaler.fit(x_train[scale_col])\n",
    "x_train[scale_col] = scaler.transform(x_train[scale_col])\n",
    "x_test[scale_col] = scaler.transform(x_test[scale_col])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "0fe2aceb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:45.921992Z",
     "iopub.status.busy": "2022-06-19T16:26:45.921568Z",
     "iopub.status.idle": "2022-06-19T16:26:46.123916Z",
     "shell.execute_reply": "2022-06-19T16:26:46.123112Z"
    },
    "papermill": {
     "duration": 0.220828,
     "end_time": "2022-06-19T16:26:46.126209",
     "exception": false,
     "start_time": "2022-06-19T16:26:45.905381",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier  \n",
    "classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  \n",
    "classifier.fit(x_train, y_train)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "8a55cfb0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.156230Z",
     "iopub.status.busy": "2022-06-19T16:26:46.155761Z",
     "iopub.status.idle": "2022-06-19T16:26:46.198999Z",
     "shell.execute_reply": "2022-06-19T16:26:46.197983Z"
    },
    "papermill": {
     "duration": 0.061147,
     "end_time": "2022-06-19T16:26:46.201583",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.140436",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_pred= classifier.predict(x_train) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "e092fce7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.231313Z",
     "iopub.status.busy": "2022-06-19T16:26:46.230865Z",
     "iopub.status.idle": "2022-06-19T16:26:46.237767Z",
     "shell.execute_reply": "2022-06-19T16:26:46.236882Z"
    },
    "papermill": {
     "duration": 0.024415,
     "end_time": "2022-06-19T16:26:46.240244",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.215829",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "83.61\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "print(round(accuracy_score(y_train, y_pred)*100,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "64079023",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.270496Z",
     "iopub.status.busy": "2022-06-19T16:26:46.270069Z",
     "iopub.status.idle": "2022-06-19T16:26:46.312279Z",
     "shell.execute_reply": "2022-06-19T16:26:46.311141Z"
    },
    "papermill": {
     "duration": 0.060293,
     "end_time": "2022-06-19T16:26:46.314758",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.254465",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "dtc = DecisionTreeClassifier()\n",
    "dtc.fit(x_train, y_train)\n",
    "y_pred = dtc.predict(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "23f54a0e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.344719Z",
     "iopub.status.busy": "2022-06-19T16:26:46.344313Z",
     "iopub.status.idle": "2022-06-19T16:26:46.350981Z",
     "shell.execute_reply": "2022-06-19T16:26:46.349958Z"
    },
    "papermill": {
     "duration": 0.024313,
     "end_time": "2022-06-19T16:26:46.353268",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.328955",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "98.2\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "print(round(accuracy_score(y_train, y_pred)*100,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d163d0e5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.384061Z",
     "iopub.status.busy": "2022-06-19T16:26:46.383635Z",
     "iopub.status.idle": "2022-06-19T16:26:46.398192Z",
     "shell.execute_reply": "2022-06-19T16:26:46.397410Z"
    },
    "papermill": {
     "duration": 0.03264,
     "end_time": "2022-06-19T16:26:46.400666",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.368026",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 418 entries, 0 to 417\n",
      "Data columns (total 16 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Pclass       418 non-null    float64\n",
      " 2   Name         418 non-null    object \n",
      " 3   Sex          418 non-null    object \n",
      " 4   Age          418 non-null    float64\n",
      " 5   SibSp        418 non-null    int64  \n",
      " 6   Parch        418 non-null    int64  \n",
      " 7   Ticket       418 non-null    object \n",
      " 8   Fare         418 non-null    float64\n",
      " 9   Cabin        91 non-null     object \n",
      " 10  Embarked     418 non-null    object \n",
      " 11  Title        418 non-null    object \n",
      " 12  Sex_en       418 non-null    int64  \n",
      " 13  Cabin_en     418 non-null    int64  \n",
      " 14  Embarked_en  418 non-null    int64  \n",
      " 15  Family_size  418 non-null    float64\n",
      "dtypes: float64(4), int64(6), object(6)\n",
      "memory usage: 55.5+ KB\n"
     ]
    }
   ],
   "source": [
    "x_test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e0e705cf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.431010Z",
     "iopub.status.busy": "2022-06-19T16:26:46.430576Z",
     "iopub.status.idle": "2022-06-19T16:26:46.436805Z",
     "shell.execute_reply": "2022-06-19T16:26:46.436126Z"
    },
    "papermill": {
     "duration": 0.023513,
     "end_time": "2022-06-19T16:26:46.438678",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.415165",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_test = x_test.drop(['Name','PassengerId','Ticket','Cabin','Embarked','Title','Sex','SibSp','Parch'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "38bac35a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.468839Z",
     "iopub.status.busy": "2022-06-19T16:26:46.468460Z",
     "iopub.status.idle": "2022-06-19T16:26:46.486118Z",
     "shell.execute_reply": "2022-06-19T16:26:46.485345Z"
    },
    "papermill": {
     "duration": 0.034987,
     "end_time": "2022-06-19T16:26:46.488037",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.453050",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_en</th>\n",
       "      <th>Cabin_en</th>\n",
       "      <th>Embarked_en</th>\n",
       "      <th>Family_size</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.4250</td>\n",
       "      <td>0.015282</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.5875</td>\n",
       "      <td>0.013663</td>\n",
       "      <td>0</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.5</td>\n",
       "      <td>0.7750</td>\n",
       "      <td>0.018909</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.3375</td>\n",
       "      <td>0.016908</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.2750</td>\n",
       "      <td>0.023984</td>\n",
       "      <td>0</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.3750</td>\n",
       "      <td>0.015713</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.4875</td>\n",
       "      <td>0.212559</td>\n",
       "      <td>0</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.4375</td>\n",
       "      <td>0.014151</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.3750</td>\n",
       "      <td>0.015713</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.3750</td>\n",
       "      <td>0.043640</td>\n",
       "      <td>1</td>\n",
       "      <td>186</td>\n",
       "      <td>0</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass     Age      Fare  Sex_en  Cabin_en  Embarked_en  Family_size\n",
       "0       1.0  0.4250  0.015282       1       186            1          0.0\n",
       "1       1.0  0.5875  0.013663       0       186            2          0.1\n",
       "2       0.5  0.7750  0.018909       1       186            1          0.0\n",
       "3       1.0  0.3375  0.016908       1       186            2          0.0\n",
       "4       1.0  0.2750  0.023984       0       186            2          0.2\n",
       "..      ...     ...       ...     ...       ...          ...          ...\n",
       "413     1.0  0.3750  0.015713       1       186            2          0.0\n",
       "414     0.0  0.4875  0.212559       0        64            0          0.0\n",
       "415     1.0  0.4375  0.014151       1       186            2          0.0\n",
       "416     1.0  0.3750  0.015713       1       186            2          0.0\n",
       "417     1.0  0.3750  0.043640       1       186            0          0.2\n",
       "\n",
       "[418 rows x 7 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "85ffe65a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.519075Z",
     "iopub.status.busy": "2022-06-19T16:26:46.518658Z",
     "iopub.status.idle": "2022-06-19T16:26:46.544421Z",
     "shell.execute_reply": "2022-06-19T16:26:46.543377Z"
    },
    "papermill": {
     "duration": 0.043951,
     "end_time": "2022-06-19T16:26:46.546791",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.502840",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_pred= classifier.predict(x_test) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "2fa892ca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.577887Z",
     "iopub.status.busy": "2022-06-19T16:26:46.577466Z",
     "iopub.status.idle": "2022-06-19T16:26:46.587936Z",
     "shell.execute_reply": "2022-06-19T16:26:46.587137Z"
    },
    "papermill": {
     "duration": 0.02888,
     "end_time": "2022-06-19T16:26:46.590230",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.561350",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "final_submission = df_test\n",
    "y_pred = pd.Series(y_pred)\n",
    "final_submission['Survived']= y_pred\n",
    "final_submission[['PassengerId','Survived']].to_csv('Submission.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "f0724131",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.621515Z",
     "iopub.status.busy": "2022-06-19T16:26:46.621069Z",
     "iopub.status.idle": "2022-06-19T16:26:46.634907Z",
     "shell.execute_reply": "2022-06-19T16:26:46.633543Z"
    },
    "papermill": {
     "duration": 0.031962,
     "end_time": "2022-06-19T16:26:46.636999",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.605037",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 891 entries, 0 to 890\n",
      "Data columns (total 7 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   Pclass       891 non-null    float64\n",
      " 1   Age          891 non-null    float64\n",
      " 2   Fare         891 non-null    float64\n",
      " 3   Sex_en       891 non-null    int64  \n",
      " 4   Cabin_en     891 non-null    int64  \n",
      " 5   Embarked_en  891 non-null    int64  \n",
      " 6   Family_size  891 non-null    float64\n",
      "dtypes: float64(4), int64(3)\n",
      "memory usage: 88.0 KB\n"
     ]
    }
   ],
   "source": [
    "x_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "76f57f36",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-06-19T16:26:46.668056Z",
     "iopub.status.busy": "2022-06-19T16:26:46.667622Z",
     "iopub.status.idle": "2022-06-19T16:26:46.996548Z",
     "shell.execute_reply": "2022-06-19T16:26:46.995517Z"
    },
    "papermill": {
     "duration": 0.347132,
     "end_time": "2022-06-19T16:26:46.998787",
     "exception": false,
     "start_time": "2022-06-19T16:26:46.651655",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "RFC = RandomForestClassifier()\n",
    "RFC.fit(x_train, y_train)\n",
    "y_pred = RFC.predict(x_test)\n",
    "final_submission['Survived']= y_pred\n",
    "final_submission[['PassengerId','Survived']].to_csv('Submission.csv',index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6200134",
   "metadata": {
    "papermill": {
     "duration": 0.014713,
     "end_time": "2022-06-19T16:26:47.028323",
     "exception": false,
     "start_time": "2022-06-19T16:26:47.013610",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 16.096438,
   "end_time": "2022-06-19T16:26:47.864966",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-06-19T16:26:31.768528",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

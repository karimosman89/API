from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from hashlib import sha256
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet

app = FastAPI()

# 
json_file_path = os.path.expanduser("C:/Users/lenov/OneDrive/Bureau/mc_reg_api/users.json")

SECRET_KEY = Fernet.generate_key()
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


class User(BaseModel):
    username: str
    first_name: str
    last_name: str
    password: str

class UserOut(BaseModel):
    username: str
    first_name: str
    last_name: str

class UserInDB(User):
    password: str 

    class Config:
        orm_mode = True

class UserPred(BaseModel):
    age: int
    sex: str
    favorite_color: str
    favorite_food: str

class Token(BaseModel):
    access_token: str
    token_type: str



model = joblib.load("model_fin2.pkl")
encoder = OneHotEncoder(categories="auto")
allowed_favorite_colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple']
allowed_favorite_foods = ['Pizza', 'Pasta', 'Burger', 'Sushi', 'Salad', 'Ice Cream']
allowed_sex = ['Male', 'Female']
encoder.fit([[color] for color in allowed_favorite_colors] + [[food] for food in allowed_favorite_foods] + [[sex] for sex in allowed_sex])



def verify_password(plain_password, hashed_password):
    return sha256(plain_password.encode()).hexdigest() == hashed_password

def load_users():
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            users_data = json.load(file)
        return [UserInDB(**user) for user in users_data]
    return []

def save_user(user: UserInDB):
    users = load_users()
    users.append(user)
    with open(json_file_path, "w") as file:
        json.dump([user.dict() for user in users], file)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")

################# Defining the endpoints

@app.post("/register", response_model=UserOut)
async def register(user: User):
    user.password = sha256(user.password.encode()).hexdigest()
    save_user(UserInDB(**user.dict(), hashed_password=user.password))
    return UserOut(**user.dict(exclude={"password"}))


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    users = load_users()

    for user in users:
        if user.username == username and verify_password(password, user.password): 
            token_data = {"sub": username}
            access_token = create_access_token(token_data, expires_delta=timedelta(minutes=30))
            return {"access_token": access_token, "token_type": "bearer"}

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

@app.post("/predict/")
async def predict_sign(
    user: UserPred,
    current_user: UserOut = Depends(get_current_user)
):
    if user.favorite_color not in allowed_favorite_colors:
        raise HTTPException(status_code=400, detail="Invalid favorite color")

    if user.favorite_food not in allowed_favorite_foods:
        raise HTTPException(status_code=400, detail="Invalid favorite food")

    if user.sex not in allowed_sex:
        raise HTTPException(status_code=400, detail="Invalid sex")

    categorical_features = [user.sex, user.favorite_color, user.favorite_food]
    encoded_features = encoder.transform([[category] for category in categorical_features]).toarray()[0]
    features = np.concatenate(([user.age], encoded_features))
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]

    return {"astrological_sign": prediction}
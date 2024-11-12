from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel

app = FastAPI()

class Salarystuff(BaseModel):
    salary: float
    bonus: float
    taxes: float

@app.get("/")
def greet():
    return {"message": "Hello handsome! Would you like to calculate your salary?"}


@app.post("/multiply/")
def multiply(number: float):
    return {f"The number multiplied by 2 is: {number * 2}"}

@app.post("/calculate/")
def calculate_salary(data: Salarystuff):
    # Check if any fields are missing
    required_fields = ['salary', 'bonus', 'taxes']
    missing_fields = [field for field in required_fields if field not in data.dict()]
    if missing_fields:
        raise HTTPException(status_code=400, detail=f"3 fields expected (salary, bonus, taxes). You forgot: {', '.join(missing_fields)}.")
    # Check if all fields are numbers
    try:
        salary = float(data.salary)
        bonus = float(data.bonus)
        taxes = float(data.taxes)
    except ValueError:
        raise HTTPException(status_code=400, detail="expected numbers, got strings.")
    # Perform the calculation
    result = salary + bonus - taxes
    return {"result": result}
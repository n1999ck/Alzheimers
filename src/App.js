import React, { useState, useEffect } from "react";
import "./App.css";
import "halfmoon/css/halfmoon.min.css";
import { useForm } from "react-hook-form";
import { DemographicsForm }  from "./DemographicsForm";

function App() {
  const { register, handleSubmit } = useForm();

  const onSubmit = (data) => {
    console.log(typeof data);
    console.log(data);
    console.log(JSON.stringify(data));
    console.log(Object.keys(data).length);
    fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);
        alert("Data has been uploaded successfully!");
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("An error occurred while uploading the data. Please try again.");
      });
  };

  return (
    <div className="App container">
      <header className="App-header">
        <form
          className="form"
          action="/api/upload"
          method="post"
          onSubmit={handleSubmit(onSubmit)}
        >
          <h3 className="mt-4">Demographics and Personal Information</h3>

          <DemographicsForm register={register} />

          <h3 className="mt-4">Lifestyle and Behavior</h3>

          {/*BMI */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              BMI:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("bmi", {
                  min: 15,
                  max: 50,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">Enter a number between 15 and 60.</div>
            </div>
          </div>

          {/*Smoking */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Smoking:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("smoking", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("smoking", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/*Alcohol Consumption */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Alcohol Consumption:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("alcoholConsumption", {
                  min: 0,
                  max: 20,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 20.
              </div>
            </div>
          </div>

          {/*Physical Activity */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Physical Activity:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("physicalActivity", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 10.
              </div>
            </div>
          </div>

          {/*Diet Quality */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Diet Quality:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("dietQuality", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 10.
              </div>
            </div>
          </div>

          {/*Sleep Quality */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Sleep Quality:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("sleepQuality", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 10.
              </div>
            </div>
          </div>

          <h3 className="mt-4">Medical History and Conditions</h3>

          {/*Family History of Alzheimer's */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Family History of Alzheimer's:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("familyHistoryAlzheimers", { required: true })}
              />
              <label className="form-check-label">Yes</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("familyHistoryAlzheimers", { required: true })}
              />
              <label className="form-check-label">No</label>
            </div>
          </div>

          {/* Cardiovascular Disease */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Cardiovascular Disease:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("cardiovascularDisease", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("cardiovascularDisease", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Diabetes */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Diabetes:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("diabetes", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("diabetes", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Depression */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Depression:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("depression", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("depression", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Head Injury */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Head Injury:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("headInjury", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("headInjury", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Hypertension */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Hypertension:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("hypertension", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("hypertension", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>
          {/* Systolic BP */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Systolic BP:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="90"
                max="179"
                className="form-control"
                {...register("systolicBP", {
                  required: true,
                  min: 90,
                  max: 179,
                })}
              />
              <div className="form-text">
                Enter an integer between 90 and 179.
              </div>
            </div>
          </div>

          {/* Diastolic BP */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Diastolic BP:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="60"
                max="119"
                className="form-control"
                {...register("diastolicBP", {
                  required: true,
                  min: 60,
                  max: 119,
                })}
              />
              <div className="form-text">
                Enter an integer between 60 and 119.
              </div>
            </div>
          </div>

          {/* Cholesterol Total */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Cholesterol Total:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="150"
                max="300"
                step="0.1"
                className="form-control"
                {...register("cholesterolTotal", {
                  required: true,
                  min: 150,
                  max: 300,
                })}
              />
              <div className="form-text">
                Enter a floating-point number between 150 and 300.
              </div>
            </div>
          </div>

          {/* Cholesterol LDL */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Cholesterol LDL:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="50"
                max="200"
                step="0.1"
                className="form-control"
                {...register("cholesterolLDL", {
                  required: true,
                  min: 50,
                  max: 200,
                })}
              />
              <div className="form-text">
                Enter a floating-point number between 50 and 200.
              </div>
            </div>
          </div>

          {/* Cholesterol HDL */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Cholesterol HDL:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="20"
                max="100"
                step="0.1"
                className="form-control"
                {...register("cholesterolHDL", {
                  required: true,
                  min: 20,
                  max: 100,
                })}
              />
              <div className="form-text">
                Enter a floating-point number between 20 and 100.
              </div>
            </div>
          </div>

          {/* Cholesterol Triglycerides */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Cholesterol Triglycerides:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                min="50"
                max="400"
                step="0.1"
                className="form-control"
                {...register("cholesterolTriglycerides", {
                  required: true,
                  min: 50,
                  max: 400,
                })}
              />
              <div className="form-text">
                Enter a floating-point number between 50 and 400.
              </div>
            </div>
          </div>

          <h3 className="mt-4">Cognitive and Functional Assessments</h3>

          {/* MMSE */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              MMSE:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("mmse", {
                  min: 0,
                  max: 30,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 30.
              </div>
            </div>
          </div>

          {/* Functional Assessment */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Functional Assessment:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                maxLength={80}
                className="form-control"
                {...register("functionalAssessment", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 10.
              </div>
            </div>
          </div>

          {/* Memory Complaints */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Memory Complaints:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("memoryComplaints", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("memoryComplaints", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Behavioral Problems */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Behavioral Problems:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("behaviorProblems", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("behaviorProblems", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* ADL */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              ADL:
            </label>
            <div className="col-sm-9">
              <input
                type="number"
                className="form-control"
                {...register("adl", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Enter a floating point number between 0 and 10.
              </div>
            </div>
          </div>

          {/* Confusion */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Confusion:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("confusion", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("confusion", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Disorientation */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Disorientation:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("disorientation", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("disorientation", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Personality Changes */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Personality Changes:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Difficulty completing tasks */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Difficulty Completing Tasks:
            </label>
            <div className="col-sm-9 form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">No</label>

              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">Yes</label>
            </div>
          </div>

          {/* Forgetfulness */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Forgetfulness:
            </label>
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("personalityChanges", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>

          <button type="submit" className="btn btn-primary mt-4">
            Submit
          </button>
        </form>
      </header>
    </div>
  );
}

export default App;

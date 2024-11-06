import React, { useState, useEffect } from "react";
import "./App.css";
import "halfmoon/css/halfmoon.min.css";
import { useForm } from "react-hook-form";
import  DemographicsForm   from "./DemographicsForm";

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

          

          <h3 className="mt-4">Medical History and Conditions</h3>

          
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
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("memoryComplaints", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("memoryComplaints", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>

          {/* Behavioral Problems */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Behavioral Problems:
            </label>
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("behavioralProblems", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("behavioralProblems", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
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
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("confusion", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("confusion", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>

          {/* Disorientation */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Disorientation:
            </label>
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("disorientation", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("disorientation", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>

          {/* Personality Changes */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Personality Changes:
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

          {/* Difficulty completing tasks */}
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
              Difficulty Completing Tasks:
            </label>
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("difficultyCompletingTasks", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("difficultyCompletingTasks", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
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
                {...register("forgetfulness", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("forgetfulness", { required: true })}
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

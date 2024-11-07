import React, { useState, useEffect } from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";
import { useForm } from "react-hook-form";
import  DemographicsForm   from "./DemographicsForm";
import  LifestyleAndBehaviorForm   from "./LifestyleAndBehaviorForm";
import  CognitiveFunctionalForm   from "./CognitiveFunctionalForm";
import  MedicalHistoryForm   from "./MedicalHistoryForm";

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
        alert(data);
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

          <LifestyleAndBehaviorForm register={register} />

          <h3 className="mt-4">Medical History and Conditions</h3>

          <MedicalHistoryForm register={register} />
          
          <h3 className="mt-4">Cognitive and Functional Assessments</h3>

          <CognitiveFunctionalForm register={register} />
          
          <button type="submit" className="btn btn-primary mt-4">
            Submit
          </button>
        </form>
      </header>
    </div>
  );
}

export default App;

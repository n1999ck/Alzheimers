import React from "react";

export default function LifestyleAndBehaviorForm({ register }) {
  return (
    <div className="section">
      {/*BMI */}
      <div className="row mb-3 mb-sm-3">
        <div className="row">
          <div className="col-sm-4">
            <label className=" col-form-label  mb-1 mb-sm-0">BMI:</label>
            <div className="">
              <input
                type="number"
                className="form-control"
                placeholder="Number between 15 and 50"
                id="bmi"
                {...register("bmi", {
                  min: 15,
                  max: 50,
                  step: 0.1,
                  required: true,
                })}
              />
            </div>
          </div>
          <div className="col-sm-4">
            {/*Smoking */}
            <div className="row mb-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">Smoking:</label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    id="smokingNo"
                    {...register("smoking", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    id="smokingYes"
                    {...register("smoking", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/*Alcohol Consumption */}
            <div className="row mb-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Alcohol Consumption:
              </label>
              <div className="">
                <input
                  type="number"
                  className="form-control"
                  placeholder="Number between 0 and 20"
                  id="alcoholConsumption"
                  {...register("alcoholConsumption", {
                    min: 0,
                    max: 20,
                    step: 0.1,
                    required: true,
                  })}
                />
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/*Physical Activity */}
            <div className="row mb-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Physical Activity:
              </label>
              <div className="">
                <input
                  type="number"
                  className="form-control"
                  placeholder="Number between 0 and 10"
                  id="physicalActivity"
                  {...register("physicalActivity", {
                    min: 0,
                    max: 10,
                    step: 0.1,
                    required: true,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/*Diet Quality */}
            <div className="row mb-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Diet Quality:
              </label>
              <div className="">
                <input
                  type="number"
                  className="form-control"
                  placeholder="Number between 0 and 10"
                  id="dietQuality"
                  {...register("dietQuality", {
                    min: 0,
                    max: 10,
                    step: 0.1,
                    required: true,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/*Sleep Quality */}
            <div className="row mb-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Sleep Quality:
              </label>
              <div className="">
                <input
                  type="number"
                  className="form-control"
                  placeholder="Number between 0 and 10"
                  id="sleepQuality"
                  {...register("sleepQuality", {
                    min: 0,
                    max: 10,
                    step: 0.1,
                    required: true,
                  })}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

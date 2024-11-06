import React from "react";

export default function LifestyleAndBehaviorForm({ register }) {
  return (
    <div>
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
            <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("smoking", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("smoking", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
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
    </div>
  )
}


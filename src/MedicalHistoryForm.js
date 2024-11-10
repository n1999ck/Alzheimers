import React from "react";

export default function CognitiveFunctionalForm({ register }) {
  return (
    <div className="section">
      <div className="row">
        <div className="row">
          <div className="col-sm-4">
            {/*Family History of Alzheimer's */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Family History of Alzheimer's:
              </label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("familyHistoryAlzheimers", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("familyHistoryAlzheimers", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Cardiovascular Disease */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label mb-1 mb-sm-0">
                Cardiovascular Disease:
              </label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("cardiovascularDisease", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("cardiovascularDisease", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Diabetes */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">Diabetes:</label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("diabetes", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("diabetes", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Depression */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Depression:
              </label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("depression", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("depression", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Head Injury */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Head Injury:
              </label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("headInjury", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("headInjury", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Hypertension */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Hypertension:
              </label>
              <div className="">
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="0"
                    {...register("hypertension", { required: true })}
                  />
                  <label className="form-check-label">No</label>
                </div>
                <div className="form-check form-check-inline">
                  <input
                    type="radio"
                    className="form-check-input"
                    value="1"
                    {...register("hypertension", { required: true })}
                  />
                  <label className="form-check-label">Yes</label>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Systolic BP */}
            <div className="row my-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">
                Systolic BP:
              </label>
              <div className="">
                <input
                  type="number"
                  min="90"
                  max="179"
                  className="form-control"
                  placeholder="Enter an integer between 90 and 179."
                  {...register("systolicBP", {
                    required: true,
                    min: 90,
                    max: 179,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Diastolic BP */}
            <div className="row my-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">
                Diastolic BP:
              </label>
              <div className="">
                <input
                  type="number"
                  min="60"
                  max="119"
                  className="form-control"
                  placeholder="Enter an integer between 60 and 119."
                  {...register("diastolicBP", {
                    required: true,
                    min: 60,
                    max: 119,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Cholesterol Total */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Cholesterol Total:
              </label>
              <div className="">
                <input
                  type="number"
                  min="150"
                  max="300"
                  step="0.1"
                  className="form-control"
                  placeholder="Enter a floating point number between 150 and 300."
                  {...register("cholesterolTotal", {
                    required: true,
                    min: 150,
                    max: 300,
                  })}
                />
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Cholesterol LDL */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Cholesterol LDL:
              </label>
              <div className="">
                <input
                  type="number"
                  min="50"
                  max="200"
                  step="0.1"
                  className="form-control"
                  placeholder="Enter a floating point number between 50 and 200."
                  {...register("cholesterolLDL", {
                    required: true,
                    min: 50,
                    max: 200,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Cholesterol HDL */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Cholesterol HDL:
              </label>
              <div className="">
                <input
                  type="number"
                  min="20"
                  max="100"
                  step="0.1"
                  className="form-control"
                  placeholder="Enter a floating point number between 20 and 100."
                  {...register("cholesterolHDL", {
                    required: true,
                    min: 20,
                    max: 100,
                  })}
                />
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Cholesterol Triglycerides */}
            <div className="row my-3 mb-sm-3">
              <label className=" col-form-label  mb-1 mb-sm-0">
                Cholesterol Triglycerides:
              </label>
              <div className="">
                <input
                  type="number"
                  min="50"
                  max="400"
                  step="0.1"
                  className="form-control"
                  placeholder="Enter a floating point number between 50 and 400."
                  {...register("cholesterolTriglycerides", {
                    required: true,
                    min: 50,
                    max: 400,
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

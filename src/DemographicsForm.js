import React from "react";

export default function DemographicsForm({ register }) {
  return (
    <div>

      {
        /* Age */
      }
      <div className="row mb-3 mb-sm-3">
        <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
          Age:
        </label>
        <div className="col-sm-9">
          <input
            type="number"
            className="form-control"
            {...register("age", { min: 0, required: true })}
          />
          <div className="form-text">Enter a whole number above 0.</div>
        </div>
      </div>

      {
        /*Gender */
      }
      <div className="row">
        <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
          Gender:
        </label>
        <div className="col-sm-9">
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="0"
                {...register("gender", { required: true })}
              />
              <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
              <input
                type="radio"
                className="form-check-input"
                value="1"
                {...register("gender", { required: true })}
              />
              <label className="form-check-label">Yes</label>
              </div>
            </div>
      </div>

      {
        /*Ethnicity */
      }
      <div className="row mb-3 mb-sm-3">
        <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
          Ethnicity:
        </label>
        <div className="col-sm-9">
          <select
            className="form-select"
            {...register("ethnicity", { required: true })}
          >
            <option value="0">Hispanic or Latino</option>
            <option value="1">White (Non-Hispanic)</option>
            <option value="2">Black or African American</option>
            <option value="3">Asian</option>
          </select>
        </div>
      </div>

      {
        /*Education Level */
      }
      <div className="row mb-3 mb-sm-3">
        <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
          Education Level:
        </label>
        <div className="col-sm-9">
          <select
            className="form-select"
            {...register("educationLevel", { required: true })}
          >
            <option value="0">Less than High School</option>
            <option value="1">High School Diploma or GED</option>
            <option value="2">Some College or Associate's Degree</option>
            <option value="3">Bachelor’s Degree or Higher</option>
          </select>
        </div>

      </div>
    </div>
  )
}


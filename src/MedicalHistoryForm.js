{/*Family History of Alzheimer's */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Family History of Alzheimer's:
</label>
<div className="col-sm-9">
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

{/* Cardiovascular Disease */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Cardiovascular Disease:
</label>
<div className="col-sm-9">
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

{/* Diabetes */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Diabetes:
</label>
<div className="col-sm-9">
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

{/* Depression */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Depression:
</label>
<div className="col-sm-9">
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

{/* Head Injury */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Head Injury:
</label>
<div className="col-sm-9">
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

{/* Hypertension */}
<div className="row mb-3 mb-sm-3">
<label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">
  Hypertension:
</label>
<div className="col-sm-9">
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

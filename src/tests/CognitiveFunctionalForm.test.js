import React from 'react';
import { render, screen } from '@testing-library/react';
import CognitiveFunctionalForm from '../components/CognitiveFunctionalForm';
import '@testing-library/jest-dom';
import { useForm } from 'react-hook-form';

describe('CognitiveFunctionalForm', () => {
  // Sample render test for CognitiveFunctionalForm
  it('renders all form fields correctly', () => {
    const TestComponent = () => {
      const { register } = useForm();
      return <CognitiveFunctionalForm register={register} />;
    };

    render(<TestComponent />);

    // Check if the family history question renders
    expect(
      screen.getByLabelText("Family History of Alzheimer's:")
    ).toBeInTheDocument();

    // Check that radio button options render for Family History
    expect(screen.getByLabelText('Yes')).toBeInTheDocument();
    expect(screen.getByLabelText('No')).toBeInTheDocument();

    // Check that cardiovascular disease question renders
    expect(
      screen.getByLabelText('Cardiovascular Disease:')
    ).toBeInTheDocument();

    // Verify presence of specific input fields like Systolic BP
    expect(screen.getByLabelText('Systolic BP:')).toBeInTheDocument();

    // Verify that helper text displays for Systolic BP
    expect(screen.getByText('Enter an integer between 90 and 179.')).toBeInTheDocument();

    // Check if Cholesterol Total field is present
    expect(screen.getByLabelText('Cholesterol Total:')).toBeInTheDocument();
  });
});

import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

test('renders demo disclosure, loading, and a returned assessment', async () => {
  let finish: (response: Response) => void = () => {};
  vi.stubGlobal('fetch', vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ demo: true, max_review_length: 10000 }) })
    .mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve; })));
  render(<App />);
  expect(await screen.findByText('Demo model active.')).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText('Your review'), 'The lamp arrived yesterday.');
  await userEvent.click(screen.getByRole('button', { name: /Analyze Review/ }));
  expect(screen.getByRole('button', { name: /Analyzing/ })).toBeDisabled();
  finish({ ok: true, json: async () => ({ label: 'genuine', authenticity_score: 72.4,
    confidence: .724, model_name: 'logistic_regression', model_version: 'test-v1',
    inference_ms: 1.5, demo: true, model_signals: [] }) } as Response);
  expect(await screen.findByText('Likely genuine')).toBeInTheDocument();
  expect(screen.getByText('Logistic Regression')).toBeInTheDocument();
  expect(screen.getByRole('meter')).toHaveAttribute('value', '72.4');
});

test('renders request errors without leaving the form disabled', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce({ ok: false })
    .mockRejectedValueOnce(new Error('Failed to fetch')));
  render(<App />);
  await userEvent.type(screen.getByLabelText('Your review'), 'A review');
  await userEvent.click(screen.getByRole('button', { name: /Analyze Review/ }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Check that the backend is running');
  expect(screen.getByRole('button', { name: /Analyze Review/ })).toBeEnabled();
});

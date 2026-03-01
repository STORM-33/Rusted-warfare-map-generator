"use client";

type LoadingScreenProps = {
  stage: string;
  progress: number;
};

export function LoadingScreen({ stage, progress }: LoadingScreenProps) {
  return (
    <div className="loading-screen">
      <h1>Loading Python runtime...</h1>
      <p>{stage}</p>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <span>{progress}%</span>
    </div>
  );
}

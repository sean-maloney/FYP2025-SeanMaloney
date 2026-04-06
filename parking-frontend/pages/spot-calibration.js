import { useRouter } from "next/router";
import SpotDrawer from "../components/spots/SpotDrawer";

export default function SpotCalibrationPage() {
  const router = useRouter();
  const { job_id, camera_id } = router.query;

  if (!job_id || !camera_id) return null;

  return <SpotDrawer jobId={job_id} cameraId={camera_id} />;
}

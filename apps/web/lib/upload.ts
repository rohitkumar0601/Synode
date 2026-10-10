import { errorDetail } from "@/components/error-state";
import { ApiError } from "@/lib/api";

/** Client-side upload guard; must equal the API's STL_MAX_UPLOAD_MB. */
export const maxUploadMb = () => Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB) || 200;

export const tooBigForUpload = (file: File | null): boolean =>
  file !== null && file.size > maxUploadMb() * 1024 * 1024;

export const tooBigMessage = () =>
  `This file is larger than the server's upload limit (${maxUploadMb()} MB), so it would be rejected.`;

/** Files above this size get a hint when the request fails without a response (status 0). */
const LARGE_UPLOAD_BYTES = 50 * 1024 * 1024;

/**
 * A status-0 failure of a large upload is most likely the server (or a proxy) refusing the body,
 * not an unreachable API: say so. Other errors pass through unchanged.
 */
export function uploadError(error: unknown, file: File | null): unknown {
  if (error instanceof ApiError && error.status === 0 && file && file.size > LARGE_UPLOAD_BYTES) {
    return new Error(`${errorDetail(error)}. A large upload may have exceeded the server's size limit.`);
  }
  return error;
}

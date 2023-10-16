import axios from "axios";

const wait = (delayMs: number) =>
  new Promise((resolve) => setTimeout(resolve, delayMs));

function isThrottleError(e: unknown) {
  return (
    (axios.isAxiosError(e) && e.response?.status === 429) ||
    (e as { status?: number }).status === 429
  );
}

/**
 * Call a provided async function and return the raw response, handling 429
 * throttling errors by retrying withexponential backoff.
 * @param reqFn Async function to call
 * @returns
 */
export async function requestWithThrottleBackoff<T>(
  reqFn: () => Promise<T>,
  maxRetries = 10
): Promise<T> {
  let numRetries = 0;
  let delay = 1;

  if (maxRetries < 0) {
    throw new Error("maxRetries must be >= 0");
  }

  while (numRetries <= maxRetries) {
    try {
      return await reqFn();
    } catch (e) {
      if (isThrottleError(e)) {
        // Too many requests
        numRetries++;

        if (numRetries > maxRetries) {
          throw new Error("Exceeded number of retry attempts");
        }

        delay *= 2 * (1 + Math.random());
        await wait(delay);
      } else {
        throw e;
      }
    }
  }

  // Shouldn't ever hit here but ts complained about missing return
  throw new Error("Exceeded number of retry attempts");
}

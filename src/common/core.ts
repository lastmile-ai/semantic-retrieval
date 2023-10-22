// Use this at the end of a switch to make it do static exhaustive checking.
export function assertUnreachable(_: never): never {
  throw new Error("Didn't expect to get here");
}

// TODO(LAS-399): Figure out proper project build structure to prevent lint error here
import { Document } from "../../src/document/document";

export async function validateFragmentChunks(
    document: Document,
    chunkSizeLimit: number,
    expectedChunks: string[]
  ) {
    expect(document.fragments.length).toEqual(expectedChunks.length);
    for (let i = 0; i < expectedChunks.length; i++) {
      const content = await document.fragments[i].getContent();
      expect(content.length).toBeLessThanOrEqual(chunkSizeLimit);
      expect(content).toEqual(expectedChunks[i]);
    }
  }
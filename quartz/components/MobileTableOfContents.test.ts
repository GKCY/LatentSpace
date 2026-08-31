import test, { describe } from "node:test"
import assert from "node:assert"
import { render } from "preact-render-to-string"
import MobileTableOfContentsConstructor from "./MobileTableOfContents"
import { QuartzComponentProps } from "./types"

const MobileTableOfContents = MobileTableOfContentsConstructor()

function makeProps(toc?: Array<{ depth: number; text: string; slug: string }>) {
  return {
    fileData: { toc },
    cfg: { locale: "zh-CN" },
  } as unknown as QuartzComponentProps
}

describe("MobileTableOfContents", () => {
  test("does not render on pages without a table of contents", () => {
    assert.strictEqual(render(MobileTableOfContents(makeProps())), "")
  })

  test("renders an accessible drawer with links to every heading", () => {
    const html = render(
      MobileTableOfContents(
        makeProps([
          { depth: 0, text: "Overview", slug: "overview" },
          { depth: 1, text: "Details", slug: "details" },
        ]),
      ),
    )

    assert.match(html, /class="mobile-toc"/)
    assert.match(html, /role="dialog"/)
    assert.match(html, /aria-controls="mobile-toc-panel"/)
    assert.match(html, /href="#overview"/)
    assert.match(html, /href="#details"/)
    assert.match(html, /data-for="details"/)
  })

  test("ships a self-contained browser script", () => {
    const script = MobileTableOfContents.afterDOMLoaded
    assert.strictEqual(typeof script, "string")
    assert.doesNotMatch(script as string, /__name/)
    assert.doesNotThrow(() => new Function(script as string))
  })
})

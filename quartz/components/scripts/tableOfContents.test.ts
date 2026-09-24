import assert from "node:assert/strict"
import test from "node:test"
import { runInNewContext } from "node:vm"
import tableOfContentsScript from "./tableOfContents"

class TocLink {
  attributes = new Map<string, string>()
  classes = new Set<string>()
  classList = {
    toggle: (name: string, active: boolean) => {
      if (active) this.classes.add(name)
      else this.classes.delete(name)
    },
  }

  constructor(
    id: string,
    readonly getBoundingClientRect: () => { top: number; bottom: number; height: number },
  ) {
    this.attributes.set("data-for", id)
  }

  getAttribute(name: string) {
    return this.attributes.get(name) ?? null
  }

  setAttribute(name: string, value: string) {
    this.attributes.set(name, value)
  }

  removeAttribute(name: string) {
    this.attributes.delete(name)
  }

  scrollIntoView() {
    assert.fail("Following the outline must not scroll ancestor containers")
  }
}

function makePage({ desktop = true, reducedMotion = false, hasDesktopToc = true } = {}) {
  let scrollY = 0
  let collapsed = false
  let frameId = 0
  const frames = new Map<number, () => void>()
  const cleanups: Array<() => void> = []
  const scrolls: ScrollToOptions[] = []
  const desktopMedia = { matches: desktop }
  const header = new EventTarget()
  const toc = {
    clientTop: 0,
    clientHeight: 200,
    scrollHeight: 640,
    scrollTop: 0,
    classList: { contains: (name: string) => name === "collapsed" && collapsed },
    getBoundingClientRect: () => ({ top: 100 }),
    querySelectorAll: () => desktopLinks,
    scrollTo: (options: ScrollToOptions) => {
      scrolls.push({ ...options })
      toc.scrollTop = options.top ?? toc.scrollTop
    },
  }
  const ids = ["intro", "basics", "example", "details", "conclusion"]
  const desktopLinks = ids.map(
    (id, index) =>
      new TocLink(id, () => {
        const top = 140 + index * 100 - toc.scrollTop
        return { top, bottom: top + 30, height: 30 }
      }),
  )
  const mobileLinks = ids.map((id) => new TocLink(id, () => ({ top: 0, bottom: 30, height: 30 })))
  const headings = ids.map((id, index) => ({
    id,
    getBoundingClientRect: () => ({ top: 200 + index * 1000 - scrollY }),
  }))
  // A heading excluded by the TOC depth limit must not clear the active parent section.
  headings.splice(2, 0, {
    id: "unlisted-subheading",
    getBoundingClientRect: () => ({ top: 1500 - scrollY }),
  })
  const document = Object.assign(new EventTarget(), {
    querySelectorAll: () => [...(hasDesktopToc ? desktopLinks : []), ...mobileLinks],
    querySelector: (selector: string) => {
      if (selector === "article") return { querySelectorAll: () => headings }
      if (!hasDesktopToc) return null
      if (selector.endsWith(".toc-content")) return toc
      if (selector.endsWith(".toc-header")) return header
      return null
    },
  })
  const window = Object.assign(new EventTarget(), {
    matchMedia: (query: string) =>
      query.includes("reduced-motion") ? { matches: reducedMotion } : desktopMedia,
    addCleanup: (cleanup: () => void) => cleanups.push(cleanup),
    scrollTo: () => assert.fail("The article viewport must not move when the outline follows"),
  })
  runInNewContext(tableOfContentsScript, {
    document,
    window,
    requestAnimationFrame: (callback: () => void) => {
      frames.set(++frameId, callback)
      return frameId
    },
    cancelAnimationFrame: (id: number) => frames.delete(id),
  })
  const flush = () => {
    const pending = [...frames.values()]
    frames.clear()
    pending.forEach((callback) => callback())
  }
  const navigate = () => document.dispatchEvent(new Event("nav"))
  navigate()
  flush()

  return {
    toc,
    desktopLinks,
    mobileLinks,
    desktopMedia,
    scrolls,
    frames,
    flush,
    navigate,
    cleanup: () => cleanups.splice(0).forEach((cleanup) => cleanup()),
    scroll: (position: number) => {
      scrollY = position
      window.dispatchEvent(new Event("scroll"))
    },
    resize: () => window.dispatchEvent(new Event("resize")),
    collapse: (value: boolean) => {
      collapsed = value
      header.dispatchEvent(new Event("click"))
    },
  }
}

function assertActive(page: ReturnType<typeof makePage>, id: string) {
  for (const links of [page.desktopLinks, page.mobileLinks]) {
    assert.deepEqual(
      links
        .filter((link) => link.classes.has("is-active"))
        .map((link) => link.getAttribute("data-for")),
      [id],
    )
    assert.equal(
      links.find((link) => link.getAttribute("data-for") === id)?.getAttribute("aria-current"),
      "location",
    )
  }
}

function assertVisible(page: ReturnType<typeof makePage>, index: number) {
  const rect = page.desktopLinks[index].getBoundingClientRect()
  assert.ok(rect.top >= 100 && rect.bottom <= 100 + page.toc.clientHeight)
}

test("follows the active section down and back up within the desktop outline", () => {
  const page = makePage()
  page.scroll(3100)
  page.flush()
  assertActive(page, "details")
  assertVisible(page, 3)
  assert.equal(page.scrolls.at(-1)?.behavior, "smooth")
  assert.ok(page.toc.scrollTop > 0)

  page.scroll(100)
  page.flush()
  assertActive(page, "intro")
  assertVisible(page, 0)
  assert.equal(page.toc.scrollTop, 0)
})

test("keeps visible entries still and lets readers browse the outline within a section", () => {
  const page = makePage()
  page.scroll(1100)
  page.flush()
  assertActive(page, "basics")
  assert.equal(page.scrolls.length, 0)

  page.toc.scrollTop = 400
  page.scroll(1400)
  page.flush()
  assertActive(page, "basics")
  assert.equal(page.scrolls.length, 0)
  assert.equal(page.toc.scrollTop, 400)
})

test("respects reduced motion when revealing an offscreen section", () => {
  const page = makePage({ reducedMotion: true })
  page.scroll(4100)
  page.flush()
  assertVisible(page, 4)
  assert.equal(page.scrolls.at(-1)?.behavior, "instant")
})

test("updates mobile highlighting without scrolling the hidden desktop outline", () => {
  const page = makePage({ desktop: false })
  page.scroll(3100)
  page.flush()
  assertActive(page, "details")
  assert.equal(page.scrolls.length, 0)

  page.desktopMedia.matches = true
  page.resize()
  page.flush()
  assertVisible(page, 3)
})

test("keeps mobile highlighting working on pages without a desktop outline", () => {
  const page = makePage({ hasDesktopToc: false })
  page.scroll(3100)
  page.flush()
  assert.equal(page.mobileLinks[3].getAttribute("aria-current"), "location")
  assert.equal(page.scrolls.length, 0)
})

test("reveals the current section when the desktop outline is reopened or resized", () => {
  const page = makePage()
  page.collapse(true)
  page.scroll(3100)
  page.flush()
  assert.equal(page.scrolls.length, 0)
  page.collapse(false)
  page.flush()
  assertVisible(page, 3)

  page.toc.clientHeight = 60
  page.resize()
  page.flush()
  assertVisible(page, 3)
})

test("batches scroll events and cleans up pending work and listeners on SPA navigation", () => {
  const page = makePage()
  page.scroll(2100)
  page.scroll(3100)
  assert.equal(page.frames.size, 1)
  page.cleanup()
  assert.equal(page.frames.size, 0)
  page.scroll(4100)
  page.resize()
  page.collapse(false)
  assert.equal(page.frames.size, 0)

  page.navigate()
  page.flush()
  assertActive(page, "conclusion")
  assertVisible(page, 4)
  assert.equal(page.scrolls.length, 1)
})

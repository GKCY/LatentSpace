import assert from "node:assert/strict"
import test from "node:test"
import { runInNewContext } from "node:vm"
import { render } from "preact-render-to-string"
import { DefaultFrame } from "./frames/DefaultFrame"
import { PageFrameProps } from "./frames/types"
import leftSidebarScript from "./scripts/leftSidebar"

function makePage({ storage = new Map<string, string>(), storageUnavailable = false } = {}) {
  const attributes = new Map<string, string>()
  const rootAttributes = new Map<string, string>()
  const label = { textContent: "收起侧栏" }
  const toggle = Object.assign(new EventTarget(), {
    setAttribute: (name: string, value: string) => attributes.set(name, value),
    querySelector: () => label,
  })
  const sidebar = { inert: false }
  const mobile = Object.assign(new EventTarget(), { matches: false })
  const cleanups: Array<() => void> = []
  let hasSidebar = true
  const document = Object.assign(new EventTarget(), {
    documentElement: {
      setAttribute: (name: string, value: string) => rootAttributes.set(name, value),
    },
    querySelector: () => (hasSidebar ? toggle : null),
    getElementById: () => (hasSidebar ? sidebar : null),
  })

  runInNewContext(leftSidebarScript, {
    document,
    window: {
      matchMedia: () => mobile,
      addCleanup: (cleanup: () => void) => cleanups.push(cleanup),
    },
    localStorage: {
      getItem: (key: string) => {
        if (storageUnavailable) throw new Error("Storage is unavailable")
        return storage.get(key) ?? null
      },
      setItem: (key: string, value: string) => {
        if (storageUnavailable) throw new Error("Storage is unavailable")
        storage.set(key, value)
      },
    },
  })

  const cleanup = () => cleanups.splice(0).forEach((callback) => callback())
  const navigate = (withSidebar = true) => {
    cleanup()
    hasSidebar = withSidebar
    document.dispatchEvent(new Event("nav"))
  }
  navigate()

  return {
    attributes,
    label,
    sidebar,
    storage,
    cleanup,
    navigate,
    collapsed: () => rootAttributes.get("data-left-sidebar-collapsed") === "true",
    click: () => toggle.dispatchEvent(new Event("click")),
    resize: (isMobile: boolean) => {
      mobile.matches = isMobile
      mobile.dispatchEvent(new Event("change"))
    },
  }
}

test("renders a reachable toggle outside the controlled sidebar only when it has content", () => {
  const props = {
    componentData: { fileData: {}, cfg: { locale: "zh-CN" } },
    header: [],
    beforeBody: [],
    pageBody: () => null,
    afterBody: [],
    left: [() => "Navigation"],
    right: [],
    footer: [],
  } as unknown as PageFrameProps
  const html = render(DefaultFrame.render(props))
  assert.match(html, /aria-controls="left-sidebar"/)
  assert.match(html, /aria-expanded="true"/)
  assert.ok(html.indexOf("</button>") < html.indexOf('id="left-sidebar"'))
  assert.doesNotMatch(
    render(DefaultFrame.render({ ...props, left: [] })),
    /data-left-sidebar-toggle/,
  )
  assert.ok(
    DefaultFrame.components?.some((component) => component.afterDOMLoaded === leftSidebarScript),
  )
})

test("collapses and restores the sidebar with accessible labels and persists across reloads", () => {
  const page = makePage()
  assert.equal(page.collapsed(), false)
  page.click()
  assert.equal(page.collapsed(), true)
  assert.equal(page.sidebar.inert, true)
  assert.equal(page.attributes.get("aria-expanded"), "false")
  assert.equal(page.attributes.get("aria-label"), "展开左侧栏")
  assert.equal(page.label.textContent, "展开侧栏")

  const reloaded = makePage({ storage: page.storage })
  assert.equal(reloaded.collapsed(), true)
  reloaded.click()
  assert.equal(reloaded.collapsed(), false)
  assert.equal(reloaded.sidebar.inert, false)
  assert.equal(reloaded.attributes.get("aria-expanded"), "true")
  assert.equal(reloaded.label.textContent, "收起侧栏")
  assert.equal(page.storage.get("left-sidebar-collapsed"), "false")
})

test("keeps mobile navigation available and restores the desktop preference on resize", () => {
  const page = makePage({ storage: new Map([["left-sidebar-collapsed", "true"]]) })
  page.resize(true)
  assert.equal(page.collapsed(), false)
  assert.equal(page.sidebar.inert, false)
  assert.equal(page.attributes.get("aria-expanded"), "true")
  assert.equal(page.storage.get("left-sidebar-collapsed"), "true")
  page.resize(false)
  assert.equal(page.collapsed(), true)
  assert.equal(page.sidebar.inert, true)
})

test("cleans up SPA listeners and preserves the preference across pages with and without sidebars", () => {
  const page = makePage()
  page.click()
  page.navigate()
  assert.equal(page.collapsed(), true)
  page.click()
  assert.equal(page.collapsed(), false)
  page.click()
  page.navigate(false)
  assert.equal(page.collapsed(), false)
  page.navigate()
  assert.equal(page.collapsed(), true)
  page.cleanup()
  page.click()
  page.resize(true)
  assert.equal(page.collapsed(), true)
})

test("still toggles when browser storage is unavailable", () => {
  const page = makePage({ storageUnavailable: true })
  page.click()
  assert.equal(page.collapsed(), true)
  page.navigate()
  assert.equal(page.collapsed(), true)
  page.click()
  assert.equal(page.collapsed(), false)
})

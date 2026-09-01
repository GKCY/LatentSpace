import assert from "node:assert/strict"
import test from "node:test"
import type { QuartzComponentProps } from "../../components/types"
import { getCondition } from "./conditions"

function propsForSlug(slug: string): QuartzComponentProps {
  return {
    fileData: { slug },
  } as QuartzComponentProps
}

test("index condition only matches the site index", () => {
  const condition = getCondition("index")
  assert.ok(condition)
  assert.equal(condition(propsForSlug("index")), true)
  assert.equal(condition(propsForSlug("foundations/RL/index")), false)
  assert.equal(condition(propsForSlug("foundations/RL/PPO")), false)
})

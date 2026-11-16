//! Local harness memory uses the session's canonical artifact directory, not
//! ambient session env. Exercise real kernels and resume without any model calls.

use std::path::{Path, PathBuf};

use pa_core::kernel::manager::ReplKernelManager;
use pa_core::kernel::shared::{ExecuteOptions, ExecuteStatus, HostRequestHandlers};
use pa_core::session_engine::{harness_digest, runtime_wiring};
use serde_json::{json, Value};

struct EnvOverride(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvOverride {
    fn apply(values: &[(&'static str, PathBuf)]) -> Self {
        Self(
            values
                .iter()
                .map(|(key, value)| {
                    let previous = std::env::var_os(key);
                    std::env::set_var(key, value);
                    (*key, previous)
                })
                .collect(),
        )
    }
}

impl Drop for EnvOverride {
    fn drop(&mut self) {
        for (key, previous) in &self.0 {
            if let Some(previous) = previous {
                std::env::set_var(key, previous);
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

fn kernel_python() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("PA_CORE_KERNEL_PYTHON") {
        let explicit = PathBuf::from(explicit);
        assert!(explicit.exists(), "PA_CORE_KERNEL_PYTHON not found");
        return Some(explicit);
    }
    let candidate =
        PathBuf::from(std::env::var_os("HOME")?).join(".prime/agent/kernel-venv/bin/python");
    if candidate.exists() {
        return Some(candidate);
    }
    eprintln!("kernel python not found; skipping live harness-memory test");
    None
}

async fn execute_json(manager: &ReplKernelManager, code: &str) -> Value {
    let result = manager
        .execute(code, ExecuteOptions::default())
        .await
        .expect("execute");
    assert_eq!(result.status, ExecuteStatus::Ok, "{result:?}");
    serde_json::from_str(result.stdout.trim()).expect("cell JSON")
}

fn provisioner(
    cwd: &Path,
    agent_dir: &Path,
    artifact_dir: Option<PathBuf>,
) -> std::sync::Arc<pa_core::kernel::provisioner::IpythonKernelProvisioner> {
    runtime_wiring::kernel_provisioner(
        "harness-memory-test".to_string(),
        HostRequestHandlers::new(),
        Vec::new(),
        cwd.to_path_buf(),
        agent_dir,
        artifact_dir,
        None,
        None,
        None,
        None,
    )
}

#[tokio::test]
#[allow(clippy::too_many_lines)] // One isolated kernel lifecycle, including resume and scope checks.
async fn local_and_global_memory_keep_their_scope_across_resume() {
    let Some(python) = kernel_python() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let root = dir.path().canonicalize().expect("canonical temp dir");
    let agent_dir = root.join("agent");
    let sessions_dir = root.join("persisted/sessions");
    let log = sessions_dir.join("session-one.jsonl");
    let artifacts = harness_digest::session_artifact_dir_for_log(&log).expect("artifact dir");
    let local_file = artifacts.join("harness/harness_state.json");
    let global_file = agent_dir.join("harness/harness_state.json");
    let stale = root.join("stale");
    let _env = EnvOverride::apply(&[
        ("PRIME_AGENT_KERNEL_PYTHON", python),
        ("RLM_SESSION_DIR", stale.join("session")),
        ("RLM_HARNESS_STATE_DIR", stale.join("local")),
        ("RLM_GLOBAL_HARNESS_STATE_DIR", stale.join("global")),
    ]);

    let first = provisioner(&root, &agent_dir, Some(artifacts.clone()));
    let manager = first.ensure(None, None).await.expect("first kernel");
    let written = execute_json(
        &manager,
        r#"
import json
import os
local = rlm.harness.create_memory("Local", "session one", id="local_note")
global_note = rlm.harness.create_memory("Global", "shared", id="global_note", global_=True)
marker = "restored"
print(json.dumps({
    "session_dir": os.environ.get("RLM_SESSION_DIR"),
    "local_file": str(rlm.harness.file_path),
    "global_file": str(rlm.get_harness_state(global_=True).file_path),
    "scopes": [local.scope, global_note.scope],
    "global_in_local": rlm.harness.get("memory", "global_note") is not None,
    "local_in_global": rlm.harness.get("memory", "local_note", global_=True) is not None,
}))
"#,
    )
    .await;
    assert_eq!(
        written,
        json!({
            "session_dir": artifacts,
            "local_file": local_file,
            "global_file": global_file,
            "scopes": ["local", "global"],
            "global_in_local": false,
            "local_in_global": false,
        })
    );
    first.dispose(None).await;
    assert!(local_file.exists());
    assert!(global_file.exists());

    let resumed = provisioner(&root, &agent_dir, Some(artifacts.clone()));
    let manager = resumed.ensure(None, None).await.expect("resumed kernel");
    assert!(resumed.last_restore().is_some());
    let restored = execute_json(
        &manager,
        r#"
import json
print(json.dumps({
    "marker": marker,
    "local": rlm.harness.get("memory", "local_note").content,
    "global": rlm.harness.get("memory", "global_note", global_=True).content,
    "local_file": str(rlm.harness.file_path),
}))
"#,
    )
    .await;
    assert_eq!(
        restored,
        json!({
            "marker": "restored",
            "local": "session one",
            "global": "shared",
            "local_file": local_file,
        })
    );
    resumed.dispose(None).await;

    let other_artifacts =
        harness_digest::session_artifact_dir_for_log(&sessions_dir.join("session-two.jsonl"))
            .expect("other artifact dir");
    let other = provisioner(&root, &agent_dir, Some(other_artifacts.clone()));
    let manager = other.ensure(None, None).await.expect("other kernel");
    let isolated = execute_json(
        &manager,
        r#"
import json
rlm.harness.create_memory("Other", "session two", id="other_note")
print(json.dumps({
    "local_missing": rlm.harness.get("memory", "local_note") is None,
    "global": rlm.harness.get("memory", "global_note", global_=True).content,
    "local_file": str(rlm.harness.file_path),
}))
"#,
    )
    .await;
    assert_eq!(
        isolated,
        json!({
            "local_missing": true,
            "global": "shared",
            "local_file": other_artifacts.join("harness/harness_state.json"),
        })
    );
    other.dispose(None).await;

    let ephemeral = provisioner(&root, &agent_dir, None);
    let manager = ephemeral
        .ensure(None, None)
        .await
        .expect("ephemeral kernel");
    let unpersisted = execute_json(
        &manager,
        r#"
import json
try:
    rlm.harness.create_memory("Lost", "must fail", id="lost")
except RuntimeError:
    rejected = True
else:
    rejected = False
rlm.harness.update_memory("global_note", "Global", "still shared", global_=True)
print(json.dumps({
    "local_file": rlm.harness.file_path,
    "local_rejected": rejected,
    "global": rlm.harness.get("memory", "global_note", global_=True).content,
}))
"#,
    )
    .await;
    assert_eq!(
        unpersisted,
        json!({"local_file": null, "local_rejected": true, "global": "still shared"})
    );
    ephemeral.dispose(None).await;
    assert!(
        !stale.exists(),
        "inherited harness paths must never be used"
    );
}
